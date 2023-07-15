#include <opencv2/opencv.hpp>
#include "../include/FeatureComparator.h"
#include "../include/Utils.h"
#include "../include/SegmentFood.h"

const std::vector<int> SegmentFood::FIRST_PLATES_LABELS{1, 2, 3, 4, 5};
const std::vector<int> SegmentFood::SECOND_PLATES_LABELS{6, 7, 8, 9};
const std::vector<int> SegmentFood::SIDE_DISHES_LABELS{10, 11};
const std::string SegmentFood::LABEL_NAMES[] = {
            "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
            "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
            "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

void SegmentFood::getFoodMaskFromPlate(cv::Mat src, cv::Mat &mask, cv::Vec3f plate) {

    cv::Point center(plate[0], plate[1]);
    int radius = plate[2];

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5,5), 0);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    // Find food mask
    mask = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    for (int r = std::max(0, center.y - radius); r < std::min(center.y + radius + 1, img.rows); r++)
        for (int c = std::max(0, center.x - radius); c < std::min(center.x + radius + 1, img.cols); c++) {
            cv::Point cur = cv::Point(c, r);
            if (cv::norm(cur - center) <= radius) {

                // Check if current point is not part of the plate
                int hsv[3] = {int(img.at<cv::Vec3b>(cur)[0]), int(img.at<cv::Vec3b>(cur)[1]), int(img.at<cv::Vec3b>(cur)[2])};
                if (hsv[1] > 80 || (hsv[0] > 20 && hsv[0] < 25))
                    mask.at<int8_t>(cur) = 1;
            }
        }
    
    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Check if there is a big area and remove the small areas (w.r.t. the plate area)
    double plateArea = SegmentFood::PI * std::pow(radius, 2);
    bool foundBigArea = false;
    for(int i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area > plateArea * 0.07)
            foundBigArea = true;
        else if (area < plateArea * 0.005) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    std::vector<std::vector<cv::Point>> keptContours;

    // Sort contours based on area
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::contourArea(contour1) > cv::contourArea(contour2);
    });

    if (foundBigArea) {

        // If there is a big area, keep only that one
        int n = 1;
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));

    } else {
        
        // Otherwise, keep the two biggest if they are not too far from the center (w.r.t. the plate radius)
        int n = 2;
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));
        for(int i = 0; i < keptContours.size(); i++) {
            double distance = std::abs(cv::pointPolygonTest(keptContours[i], center, true));
            // std::cout << "Distance " << distance << std::endl;
            if (distance > radius * 0.75) {
                keptContours.erase(keptContours.begin() + i);
                i--;
            }
        }
    }

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, keptContours, -1, cv::Scalar(1), cv::FILLED);

    // Fill the holes
    int closingSize = radius / 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

void SegmentFood::getFoodMaskFromPlates(
    cv::Mat src, cv::Mat &mask, std::vector<cv::Vec3f> plates, std::vector<int> labelsFound)
{

    cv::Mat segmentationMask(src.size(), CV_8UC1, cv::Scalar(0));

    cv::Mat labels;
    FeatureComparator::readLabelFeaturesFromFile(labels);

    std::vector<std::vector<FeatureComparator::LabelDistance>> platesLabelDistances;
    std::vector<int> allowedLabels;
    std::vector<cv::Mat> platesMasks;

    allowedLabels = Utils::getVectorUnion(FIRST_PLATES_LABELS, Utils::getVectorUnion(SECOND_PLATES_LABELS, SIDE_DISHES_LABELS));
    if (!labelsFound.empty())
        allowedLabels = Utils::getVectorIntersection(allowedLabels, labelsFound);

    for(int i = 0; i < plates.size(); i++) {

        cv::Mat tmpMask;
        
        SegmentFood::getFoodMaskFromPlate(src, tmpMask, plates[i]);

        // Do not consider empty plate masks
        if (cv::countNonZero(tmpMask) == 0) {
            plates.erase(plates.begin() + i);
            i--;
            continue;
        }

        platesMasks.push_back(tmpMask);
        
        // Creates the features for the segmented patch
        cv::Mat patchFeatures;
        FeatureComparator::getImageFeatures(src, tmpMask, patchFeatures);
        platesLabelDistances.push_back(FeatureComparator::getLabelDistances(labels, allowedLabels, patchFeatures));
    }

    /*
    // Print label distances
    for (int j = 0; j < platesLabelDistances.size(); j++) {
        for (int k = 0; k < platesLabelDistances[j].size(); k++)
            std::cout << platesLabelDistances[j][k].label << " " << platesLabelDistances[j][k].distance << " - ";
        std::cout << std::endl;
    }
    */    

    // Choose best labels such that if there are more plates, they are one first and one second plate
    if (platesLabelDistances.size() > 1) {

        int plate0FirstIdx = -1, plate0SecondIdx = -1, plate1FirstIdx = -1, plate1SecondIdx = -1;

        // Find best first and second plate option for plate 0
        for (int j = 0; j < platesLabelDistances[0].size(); j++) {
            if (Utils::getIndexInVector(FIRST_PLATES_LABELS, platesLabelDistances[0][j].label) != -1) {
                if (plate0FirstIdx == -1)
                    plate0FirstIdx = j;
            } else {
                if (plate0SecondIdx == -1)
                    plate0SecondIdx = j;
            }
            if (plate0FirstIdx != -1 && plate0SecondIdx != -1)
                break;
        }

        // Find best first and second plate option for plate 1
        for (int j = 0; j < platesLabelDistances[1].size(); j++) {
            if (Utils::getIndexInVector(FIRST_PLATES_LABELS, platesLabelDistances[1][j].label) != -1) {
                if (plate1FirstIdx == -1)
                    plate1FirstIdx = j;
            } else {
                if (plate1SecondIdx == -1)
                    plate1SecondIdx = j;
            }
            if (plate1FirstIdx != -1 && plate1SecondIdx != -1)
                break;
        }

        // Compute the loss between choosing the best first or second dish normalized using the worst label distance
        double plate0NormalizedLoss = std::abs(platesLabelDistances[0][plate0FirstIdx].distance - platesLabelDistances[0][plate0SecondIdx].distance)
                                        / platesLabelDistances[0][platesLabelDistances[0].size() - 1].distance;
        double plate1NormalizedLoss = std::abs(platesLabelDistances[1][plate1FirstIdx].distance - platesLabelDistances[1][plate1SecondIdx].distance)
                                        / platesLabelDistances[1][platesLabelDistances[1].size() - 1].distance;
        
        if (plate0NormalizedLoss < plate1NormalizedLoss) {
            
            // If plate 0 has less loss, give preference to plate 1
            if (plate1FirstIdx < plate1SecondIdx) {
                platesLabelDistances[0][0] = platesLabelDistances[0][plate0SecondIdx];
                platesLabelDistances[1][0] = platesLabelDistances[1][plate1FirstIdx];
            } else {
                platesLabelDistances[0][0] = platesLabelDistances[0][plate0FirstIdx];
                platesLabelDistances[1][0] = platesLabelDistances[1][plate1SecondIdx];
            }
        } else {

            // Otherwise, give preference to plate 1
            if (plate0FirstIdx < plate0SecondIdx) {
                platesLabelDistances[0][0] = platesLabelDistances[0][plate0FirstIdx];
                platesLabelDistances[1][0] = platesLabelDistances[1][plate1SecondIdx];
            } else {
                platesLabelDistances[0][0] = platesLabelDistances[0][plate0SecondIdx];
                platesLabelDistances[1][0] = platesLabelDistances[1][plate1FirstIdx];
            }
        }
    }

    for (int i = 0; i < plates.size(); i++) {
        std::cout << "Plate " << i << std::endl;

        int foodLabel = platesLabelDistances[i][0].label;
        double labelDistance = platesLabelDistances[i][0].distance;
        
        // If it is a first plate, we have finished
        if (Utils::getIndexInVector(FIRST_PLATES_LABELS, foodLabel) != -1) {

            // Refine segmentation
            SegmentFood::refineMask(src, platesMasks[i], foodLabel);

            // Add to segmentation mask
            for(int r = 0; r < segmentationMask.rows; r++)
                for(int c = 0; c < segmentationMask.cols; c++)
                    if(platesMasks[i].at<uchar>(r,c) != 0)
                        segmentationMask.at<uchar>(r,c) = int(platesMasks[i].at<uchar>(r,c)*foodLabel);

            std::cout << "Label found: " << LABEL_NAMES[foodLabel] << std::endl;
        } else {

            // We have to split the mask into more foods
            allowedLabels = Utils::getVectorUnion(SECOND_PLATES_LABELS, SIDE_DISHES_LABELS);  
            if (!labelsFound.empty())
                allowedLabels = Utils::getVectorIntersection(labelsFound, allowedLabels);

            // Crop mask to non zero area
            cv::Mat tmpMask = platesMasks[i];
            cv::Rect bbox = cv::boundingRect(tmpMask);
            cv::Mat croppedMask = tmpMask(bbox).clone();

            // Split mask into a grid of smaller masks
            int BIG_WINDOW_SIZE = SegmentFood::BIG_WINDOW_SIZE;
            int windowSize = std::min(BIG_WINDOW_SIZE, std::min(croppedMask.rows, croppedMask.cols));
            std::vector<int> gridLabels;

            for (int y = 0; y < croppedMask.rows; y += windowSize) {
                for (int x = 0; x < croppedMask.cols; x += windowSize) {
                    
                    // Compute submask
                    cv::Rect windowRect(x, y, std::min(windowSize, croppedMask.cols - x), std::min(windowSize, croppedMask.rows - y));
                    cv::Mat submask = croppedMask(windowRect).clone();

                    if (cv::countNonZero(submask) == 0)
                        continue;

                    cv::Mat curMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(curMask(bbox)(windowRect)); 

                    /*
                    // Print each masked window
                    cv::Mat masked;
                    cv::bitwise_and(src, src, masked, curMask);
                    cv::imshow("masked", masked);
                    cv::waitKey();
                    */

                    // Find a label for each submask                  
                    cv::Mat patchFeatures;
                    FeatureComparator::getImageFeatures(src, curMask, patchFeatures);
                    foodLabel = FeatureComparator::getLabelDistances(labels, allowedLabels, patchFeatures)[0].label;
                    gridLabels.push_back(foodLabel);
                }
            }

            // Repeat using only the most frequent labels of the previous step
            Utils::sortVectorByFreq(gridLabels);
            std::vector<int> foundSecondPlates = Utils::getVectorIntersection(gridLabels, SECOND_PLATES_LABELS);
            std::vector<int> foundSideDishes = Utils::getVectorIntersection(gridLabels, SIDE_DISHES_LABELS);

            std::vector<int> keptLabels;
            if (!foundSecondPlates.empty())
                keptLabels.push_back(foundSecondPlates[0]);
            if (!foundSideDishes.empty())
                keptLabels.push_back(foundSideDishes[0]);
            if (foundSideDishes.size() > 1)
                keptLabels.push_back(foundSideDishes[1]);

            /*
            // Print kept labels
            std::cout << "Kept labels: ";
            for (int j = 0; j < keptLabels.size(); j++)
                std::cout << keptLabels[j] << " ";
            std::cout << std::endl;
            */

            // Use smaller submasks for a better segmentation results
            int SMALL_WINDOW_SIZE = SegmentFood::SMALL_WINDOW_SIZE;
            windowSize = std::min(SMALL_WINDOW_SIZE, std::min(croppedMask.rows, croppedMask.cols));

            // Create a matrix of the assigned labels
            int rows = std::ceil(float(croppedMask.rows) / float(windowSize));
            int cols = std::ceil(float(croppedMask.cols) / float(windowSize));
            std::vector<std::vector<int>> labelMat(rows, std::vector<int> (cols)); 

            for (int y = 0; y < croppedMask.rows; y += windowSize) {
                
                int row = std::floor(float(y) / float(windowSize));
                for (int x = 0; x < croppedMask.cols; x += windowSize) {
                    
                    int col = std::floor(float(x) / float(windowSize));

                    // Compute submask
                    cv::Rect windowRect(x, y, std::min(windowSize, croppedMask.cols - x), std::min(windowSize, croppedMask.rows - y));
                    cv::Mat submask = croppedMask(windowRect).clone();

                    if (cv::countNonZero(submask) == 0) {
                        labelMat[row][col] = -1;
                        continue;
                    }

                    cv::Mat curMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(curMask(bbox)(windowRect));

                    // Find a label for each submask
                    cv::Mat patchFeatures;
                    FeatureComparator::getImageFeatures(src, curMask, patchFeatures);
                    foodLabel = FeatureComparator::getLabelDistances(labels, keptLabels, patchFeatures)[0].label;
                    labelMat[row][col] = foodLabel;
                }
            }

            // Re-assign labels based also on the neighboring submasks
            Utils::getMostFrequentMatrix(labelMat, 1);

            // Merge together submasks with the same label
            std::vector<cv::Mat> foodMasks(keptLabels.size());

            for (int y = 0; y < croppedMask.rows; y += windowSize) {

                int row = std::floor(float(y) / float(windowSize));
                for (int x = 0; x < croppedMask.cols; x += windowSize) {

                    int col = std::floor(float(x) / float(windowSize));

                    // Compute submask
                    cv::Rect windowRect(x, y, std::min(windowSize, croppedMask.cols - x), std::min(windowSize, croppedMask.rows - y));
                    cv::Mat submask = croppedMask(windowRect).clone();

                    if (cv::countNonZero(submask) == 0)
                        continue;

                    cv::Mat curMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(curMask(bbox)(windowRect));

                    // Get label
                    foodLabel = labelMat[row][col];
                    int index = Utils::getIndexInVector(keptLabels, foodLabel);

                    // Add to the mask of the corresponding label
                    if (foodMasks[index].empty())
                        foodMasks[index] = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    foodMasks[index] += curMask;                  
                }
            }

            // Refine segmentation re-assigning small pieces of food
            if (keptLabels.size() == 2) {
                for (int j = 0; j < keptLabels.size(); j++) {

                    if (cv::countNonZero(foodMasks[j]) == 0)
                        continue;

                    // Find contours in the mask
                    std::vector<std::vector<cv::Point>> contours, keptContours;
                    cv::findContours(foodMasks[j], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    // Sort contours based on area
                    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
                        return cv::contourArea(contour1) > cv::contourArea(contour2);
                    });

                    // Keep biggest contour if it is not too small
                    if (contours.size() > 0 && cv::contourArea(contours[0]) > 500) {
                        keptContours.push_back(contours.front());
                        contours.erase(contours.begin());

                        // Keep second biggest contour if it is slightly smaller
                        if (contours.size() > 0 && cv::contourArea(contours[0]) > cv::contourArea(keptContours[0]) * 0.90) {
                            keptContours.push_back(contours.front());
                            contours.erase(contours.begin());
                        }
                    }

                    // Draw kept countours
                    foodMasks[j] = cv::Mat::zeros(foodMasks[j].size(), CV_8UC1);
                    cv::drawContours(foodMasks[j], keptContours, -1, cv::Scalar(1), cv::FILLED);
                    
                    // Draw other countours on the other mask
                    if (j == 0)
                        cv::drawContours(foodMasks[1], contours, -1, cv::Scalar(1), cv::FILLED);
                    else
                        cv::drawContours(foodMasks[0], contours, -1, cv::Scalar(1), cv::FILLED);
                    
                }
            }

            // Add found masks to the segmentation mask
            for (int j = 0; j < keptLabels.size(); j++) {

                if (cv::countNonZero(foodMasks[j]) == 0)
                        continue;

                foodLabel = keptLabels[j];

                // Refine segmentation
                SegmentFood::refineMask(src, platesMasks[i], foodLabel);

                for(int r = 0; r < segmentationMask.rows; r++)
                        for(int c = 0; c < segmentationMask.cols; c++)
                            if(foodMasks[j].at<uchar>(r,c) != 0)
                                segmentationMask.at<uchar>(r,c) = int(foodMasks[j].at<uchar>(r,c) * foodLabel);
                
                std::cout << "Label found: " << LABEL_NAMES[foodLabel] << std::endl;
            }
        }
    }

    mask = segmentationMask;
}

void SegmentFood::getSaladMaskFromBowl(cv::Mat src, cv::Mat &mask, cv::Vec3f bowl) {

    cv::Point center(bowl[0], bowl[1]);
    int radius = bowl[2];

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5,5), 0);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    // Find salad mask
    mask = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    for (int r = std::max(0, center.y - radius); r < std::min(center.y + radius + 1, img.rows); r++)
        for (int c = std::max(0, center.x - radius); c < std::min(center.x + radius + 1, img.cols); c++) {
            cv::Point cur = cv::Point(c, r);
            if (cv::norm(cur - center) <= radius) {

                // Check if current point is not part of the bowl
                int hsv[3] = {int(img.at<cv::Vec3b>(cur)[0]), int(img.at<cv::Vec3b>(cur)[1]), int(img.at<cv::Vec3b>(cur)[2])};
                if (hsv[1] > 175 || hsv[2] > 245)
                    mask.at<int8_t>(cur) = SALAD_LABEL;
            }
        }

    // Fill the holes
    int closingSize = radius / 3;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Remove the small areas (w.r.t. the bowl area)
    double bowlArea = SegmentFood::PI * std::pow(radius, 2);
    for(int i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area < bowlArea * 0.001) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, contours, -1, cv::Scalar(SALAD_LABEL), cv::FILLED);
}

cv::Mat SegmentFood::getBreadMask(cv::Mat src, cv::Mat breadMask) {

    int kernelSizeErosion = 5; // Adjust the size according to your needs
    cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeErosion, kernelSizeErosion));
    cv::Mat erodedMask;
    cv::morphologyEx(breadMask, erodedMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 3);

    cv::Rect bbx = cv::boundingRect(erodedMask);
    if (bbx.empty())
        return cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat final_image, result_mask, bgModel, fgModel;

    // GrabCut segmentation algorithm for current box
    grabCut(src,		// input image
        result_mask,	// segmentation resulting mask
        bbx,			// rectangle containing foreground
        bgModel, fgModel,	// models
        10,				// number of iterations
        cv::GC_INIT_WITH_RECT);
    
    cv::Mat tmpMask0 = (result_mask == 0);
    cv::Mat tmpMask1 = (result_mask == 1);
    cv::Mat tmpMask2 = (result_mask == 2);
    cv::Mat tmpMask3 = (result_mask == 3);

    cv::Mat foreground = (tmpMask3 | tmpMask1);

    // Perform connected component analysis
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat out = cv::Mat::zeros(foreground.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestAreaPost) {
            largestAreaPost = area;
            index = i;
        }    
    }
    if (index != -1) 
        cv::fillPoly(out, contours[index], cv::Scalar(BREAD_LABEL));

    return out;
}   

void SegmentFood::refinePestoPasta(const cv::Mat& src, cv::Mat& mask) {
    cv::Mat workingFood;
    cv::Mat helperMask = mask.clone();
    cv::Mat fullSizeMask(src.rows, src.cols, CV_8UC1, cv::Scalar(1));

    cv::bitwise_and(src, src, workingFood, helperMask);

    //cv::imshow("tmp1", workingFood); cv::waitKey();

    cv::Mat hsvImage;
    cv::cvtColor(workingFood, hsvImage, cv::COLOR_BGR2HSV);
    
    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    cv::Mat thresholdingMask = hsvChannels[1] > 0.6*255;
    cv::Mat thresholdedImage;
    cv::bitwise_and(src, src, thresholdedImage, thresholdingMask);

    //cv::imshow("thresholdedImage", thresholdedImage);

    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> thresholdedImageChannels;
    cv::split(thresholdedImage, thresholdedImageChannels);
    cv::Mat thresholdedGreenMask = thresholdingMask;

    for (int i = 0; i < thresholdingMask.rows; i++) {
        for (int j = 0; j < thresholdingMask.cols; j++) {
            if (thresholdedImageChannels[1].at<uchar>(i,j) < 85)
                thresholdedGreenMask.at<uchar>(i,j) = 0;
        }
    }

    int kernelSizeDilation = 5;
    cv::Mat kernelDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeDilation, kernelSizeDilation));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat dilatedMask;
    cv::morphologyEx(thresholdedGreenMask, dilatedMask, cv::MORPH_DILATE, kernelDilation, cv::Point(-1, -1), 1);

    int kernelSizeClosure = 3;
    cv::Mat kernelClosure = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosure, kernelSizeClosure));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat closedMask;
    cv::morphologyEx(dilatedMask, closedMask, cv::MORPH_CLOSE, kernelClosure, cv::Point(1, 1), 10);

    int kernelSizeOpening = 3;
    cv::Mat kernelOpening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeOpening, kernelSizeOpening));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat openMask;
    cv::morphologyEx(closedMask, openMask, cv::MORPH_OPEN, kernelOpening, cv::Point(1, 1), 2);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(openMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(openMask.size(), CV_8UC1);


    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }

    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(1));
    
    mask = filledMask;
}

cv::Mat SegmentFood::refineTomatoPasta(cv::Mat src, cv::Mat mask) {
    cv::Mat workingFood;
    cv::bitwise_and(src, src, workingFood, mask);
    // cv::imshow("workingFood", workingFood);  cv::waitKey();
    
    // Split the image into individual BGR channels
    std::vector<cv::Mat> channels;
    cv::split(workingFood, channels);

    // Keep only the red channel
    cv::Mat redChannel = channels[2];

    cv::Mat thresholdedRedChannel = redChannel > 160;
    cv::Mat bgrThresholded;
    cv::bitwise_and(workingFood, workingFood, bgrThresholded, thresholdedRedChannel);
    
    //cv::imshow("thresholdedRedChannel", thresholdedRedChannel);


    int kernelSizeClosing = 3;
    cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosing, kernelSizeClosing));
    // Perform dilation followed by erosion (closing operation)
    cv::Mat closingMask;
    cv::morphologyEx(thresholdedRedChannel, closingMask, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 5);
    cv::Mat closingImage;
    cv::bitwise_and(workingFood, workingFood, closingImage, closingMask);

    //cv::imshow("closingMask", closingMask);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closingMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(closingMask.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }

    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(255));

    //cv::imshow("filledMask", filledMask);
    //cv::imshow("bgrThresholded", bgrThresholded);

    //cv::waitKey();
    return bgrThresholded ;
}

void SegmentFood::refinePorkCutlet(cv::Mat src, cv::Mat &mask) {
        
    // Close the holes borders
    int closingSize = cv::boundingRect(mask).height / 4;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Sort contours based on area
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::contourArea(contour1) > cv::contourArea(contour2);
    });

    int n = 1;
    std::vector<std::vector<cv::Point>> keptContours;
    keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, keptContours, -1, cv::Scalar(1), cv::FILLED);
}

void SegmentFood::refineMask(const cv::Mat& src, cv::Mat& mask, int label) {
    switch (label) {
        case 1:
            refinePestoPasta(src, mask);
            break;
        case 2:
            refineTomatoPasta(src, mask);
            break;
        case 6:
            refinePorkCutlet(src, mask);
            break;
        default:
            break;
    }
}