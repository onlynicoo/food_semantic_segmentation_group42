#include "../include/FoodSegmenter.h"

#include <opencv2/opencv.hpp>

#include "../include/HistogramThresholder.h"
#include "../include/Utils.h"

const std::vector<int> FoodSegmenter::FIRST_PLATES_LABELS{1, 2, 3, 4, 5};
const std::vector<int> FoodSegmenter::SECOND_PLATES_LABELS{6, 7, 8, 9};
const std::vector<int> FoodSegmenter::SIDE_DISHES_LABELS{10, 11};
const std::string FoodSegmenter::LABEL_NAMES[] = {
    "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
    "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
    "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

/**
 * The function `getFoodMaskFromPlate` takes an input image, plate coordinates, and radius, and returns
 * a binary mask of the food items on the plate.
 * 
 * @param src The source image from which the food mask is to be extracted. It should be a 3-channel
 * BGR image.
 * @param mask The "mask" parameter is an output parameter of type cv::Mat. It is used to store the
 * binary mask image that represents the segmented food regions on the plate.
 * @param plate The plate parameter is a vector of three elements representing the center coordinates
 * and radius of the plate. The first element (plate[0]) represents the x-coordinate of the center, the
 * second element (plate[1]) represents the y-coordinate of the center, and the third element
 * (plate[2])
 * @return Nothing is being returned from this function. The function is modifying the `mask` parameter
 * passed by reference.
 */
void FoodSegmenter::getFoodMaskFromPlate(const cv::Mat& src, cv::Mat& mask, cv::Vec3f plate) {
    
    cv::Point center(plate[0], plate[1]);
    int radius = plate[2];

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5, 5), 0);

    // Find food mask
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
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
    double plateArea = FoodSegmenter::PI * std::pow(radius, 2);
    bool foundBigArea = false;
    for (int i = 0; i < contours.size(); i++) {
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
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int)contours.size()));

    } else {
        // Otherwise, keep the two biggest if they are not too far from the center (w.r.t. the plate radius)
        int n = 2;
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int)contours.size()));
        for (int i = 0; i < keptContours.size(); i++) {
            double distance = std::abs(cv::pointPolygonTest(keptContours[i], center, true));
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

/**
 * The function `getFoodMaskFromPlates` takes an input image, a list of plate positions, and a list of
 * labels, and returns a mask indicating the regions of the image that contain food.
 * 
 * @param src The source image from which the food masks will be extracted. It is of type cv::Mat.
 * @param mask The `mask` parameter is an output parameter of type `cv::Mat`. It is used to store the
 * segmentation mask of the food in the input image. The segmentation mask is a binary image where the
 * pixels belonging to the food are set to non-zero values, and the pixels not belonging to the
 * @param plates A vector of cv::Vec3f representing the plates detected in the image. Each cv::Vec3f
 * contains the center coordinates (x, y) and the radius of the plate.
 * @param labelsFound The parameter "labelsFound" is a vector of integers that represents the labels of
 * the food items that have been found in the image.
 * @return Nothing is being returned. The function is void, meaning it does not return any value.
 */
void FoodSegmenter::getFoodMaskFromPlates(const cv::Mat& src, cv::Mat& mask, std::vector<cv::Vec3f> plates, std::vector<int>& labelsFound) {
    
    cv::Mat segmentationMask(src.size(), CV_8UC1, cv::Scalar(0));

    cv::Mat labelsHistograms;
    HistogramThresholder::readLabelsHistogramsFromFile(labelsHistograms);

    std::vector<std::vector<HistogramThresholder::LabelDistance>> platesLabelDistances;
    std::vector<int> allowedLabels, curLabels;
    std::vector<cv::Mat> platesMasks;

    allowedLabels = Utils::getVectorUnion(FIRST_PLATES_LABELS, Utils::getVectorUnion(SECOND_PLATES_LABELS, SIDE_DISHES_LABELS));
    if (!labelsFound.empty())
        allowedLabels = Utils::getVectorIntersection(allowedLabels, labelsFound);

    for (int i = 0; i < plates.size(); i++) {
        cv::Mat tmpMask;

        FoodSegmenter::getFoodMaskFromPlate(src, tmpMask, plates[i]);

        // Do not consider empty plate masks
        if (cv::countNonZero(tmpMask) == 0) {
            plates.erase(plates.begin() + i);
            i--;
            continue;
        }

        platesMasks.push_back(tmpMask);

        // Evaluates the histogram for the segmented patch
        cv::Mat patchHistogram;
        HistogramThresholder::getImageHistogram(src, tmpMask, patchHistogram);
        platesLabelDistances.push_back(HistogramThresholder::getLabelDistances(labelsHistograms, allowedLabels, patchHistogram));
    }

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

        int foodLabel = platesLabelDistances[i][0].label;
        double labelDistance = platesLabelDistances[i][0].distance;

        // If it is a first plate, we have finished
        if (Utils::getIndexInVector(FIRST_PLATES_LABELS, foodLabel) != -1) {

            // Refine segmentation
            FoodSegmenter::refineMask(src, platesMasks[i], foodLabel);

            // Add to segmentation mask
            for (int r = 0; r < segmentationMask.rows; r++)
                for (int c = 0; c < segmentationMask.cols; c++)
                    if (platesMasks[i].at<uchar>(r, c) != 0)
                        segmentationMask.at<uchar>(r, c) = int(platesMasks[i].at<uchar>(r, c) * foodLabel);

            curLabels.push_back(foodLabel);

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
            int BIG_WINDOW_SIZE = FoodSegmenter::BIG_WINDOW_SIZE;
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

                    // Find a label for each submask
                    cv::Mat patchHistogram;
                    HistogramThresholder::getImageHistogram(src, curMask, patchHistogram);
                    foodLabel = HistogramThresholder::getLabelDistances(labelsHistograms, allowedLabels, patchHistogram)[0].label;
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

            // Use smaller submasks for a better segmentation results
            int SMALL_WINDOW_SIZE = FoodSegmenter::SMALL_WINDOW_SIZE;
            windowSize = std::min(SMALL_WINDOW_SIZE, std::min(croppedMask.rows, croppedMask.cols));

            // Create a matrix of the assigned labels
            int rows = std::ceil(float(croppedMask.rows) / float(windowSize));
            int cols = std::ceil(float(croppedMask.cols) / float(windowSize));
            std::vector<std::vector<int>> labelMat(rows, std::vector<int>(cols));

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
                    cv::Mat patchHistogram;
                    HistogramThresholder::getImageHistogram(src, curMask, patchHistogram);
                    foodLabel = HistogramThresholder::getLabelDistances(labelsHistograms, keptLabels, patchHistogram)[0].label;
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
                FoodSegmenter::refineMask(src, foodMasks[j], foodLabel);

                for (int r = 0; r < segmentationMask.rows; r++)
                    for (int c = 0; c < segmentationMask.cols; c++)
                        if (foodMasks[j].at<uchar>(r, c) != 0)
                            segmentationMask.at<uchar>(r, c) = int(foodMasks[j].at<uchar>(r, c) * foodLabel);

                curLabels.push_back(foodLabel);
            }
        }
    }

    if (labelsFound.empty())
        labelsFound = curLabels;

    mask = segmentationMask;
}

/**
 * The function `getSaladMaskFromBowl` takes an input image, a bowl position, and returns a binary mask
 * of the salad in the image.
 * 
 * @param src The source image from which the salad mask will be extracted. It should be a 3-channel
 * BGR image.
 * @param mask The `mask` parameter is an output parameter of type `cv::Mat&`, which is a reference to
 * a `cv::Mat` object. It is used to store the resulting salad mask.
 * @param bowl The bowl parameter is a vector of three elements (cv::Vec3f) representing the center
 * coordinates (x, y) and the radius of the bowl.
 */
void FoodSegmenter::getSaladMaskFromBowl(const cv::Mat& src, cv::Mat& mask, cv::Vec3f bowl) {

    cv::Point center(bowl[0], bowl[1]);
    int radius = bowl[2];

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5, 5), 0);
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
    double bowlArea = FoodSegmenter::PI * std::pow(radius, 2);
    for (int i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area < bowlArea * 0.001) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, contours, -1, cv::Scalar(SALAD_LABEL), cv::FILLED);
}

/**
 * The function `getBreadMask` takes an input image and a bread area mask, and generates a binary mask
 * that represents the bread region in the image.
 * 
 * @param src The input image on which the segmentation will be performed.
 * @param breadArea The `breadArea` parameter is a binary image that represents the area where the
 * bread is located in the input image (`src`). It is used as a mask to segment the bread from the rest
 * of the image.
 * @param breadMask The `breadMask` parameter is an output parameter of type `cv::Mat`. It is used to
 * store the binary mask of the bread area in the input image. The mask will have the same size as the
 * input image and will contain white pixels (255) where the bread is present and black
 * @return The function does not return a value. It modifies the `breadMask` parameter, which is passed
 * by reference.
 */
void FoodSegmenter::getBreadMask(const cv::Mat& src, const cv::Mat& breadArea, cv::Mat& breadMask) {
    int kernelSizeErosion = 5;  // Adjust the size according to your needs
    cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeErosion, kernelSizeErosion));
    cv::Mat erodedMask;
    cv::morphologyEx(breadArea, erodedMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 3);

    cv::Rect bbx = cv::boundingRect(erodedMask);
    
    if (bbx.empty()) {
        breadMask = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
        return;
    }

    cv::Mat final_image, result_mask, bgModel, fgModel;

    // GrabCut segmentation algorithm for current box
    grabCut(src,               // input image
            result_mask,       // segmentation resulting mask
            bbx,               // rectangle containing foreground
            bgModel, fgModel,  // models
            3,                // number of iterations
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
    breadMask = cv::Mat::zeros(foreground.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestAreaPost) {
            largestAreaPost = area;
            index = i;
        }
    }
    if (index != -1)
        cv::fillPoly(breadMask, contours[index], cv::Scalar(BREAD_LABEL));
}

/**
 * The function refines a given mask for identifying pesto pasta in an image by applying various image
 * processing techniques.
 * 
 * @param src The `src` parameter is a `cv::Mat` object representing the source image. It is used to
 * perform various image processing operations.
 * @param mask The "mask" parameter is a binary image that represents the region of interest in the
 * source image. It is passed by reference and will be modified by the function to contain the refined
 * mask after segmentation.
 */
void FoodSegmenter::refinePestoPasta(const cv::Mat& src, cv::Mat& mask) {
    cv::Mat workingFood;
    cv::Mat helperMask = mask.clone();
    cv::Mat fullSizeMask(src.rows, src.cols, CV_8UC1, cv::Scalar(1));

    cv::bitwise_and(src, src, workingFood, helperMask);

    // cv::imshow("tmp1", workingFood); cv::waitKey();

    cv::Mat hsvImage;
    cv::cvtColor(workingFood, hsvImage, cv::COLOR_BGR2HSV);

    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    cv::Mat thresholdingMask = hsvChannels[1] > 0.6 * 255;
    cv::Mat thresholdedImage;
    cv::bitwise_and(src, src, thresholdedImage, thresholdingMask);

    // cv::imshow("thresholdedImage", thresholdedImage);

    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> thresholdedImageChannels;
    cv::split(thresholdedImage, thresholdedImageChannels);
    cv::Mat thresholdedGreenMask = thresholdingMask;

    for (int i = 0; i < thresholdingMask.rows; i++) {
        for (int j = 0; j < thresholdingMask.cols; j++) {
            if (thresholdedImageChannels[1].at<uchar>(i, j) < 85)
                thresholdedGreenMask.at<uchar>(i, j) = 0;
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
    for (int i = 0; i < contours.size(); i++) {
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

/**
 * The function refines a given mask by applying various image processing techniques to segment and
 * fill the contours of the food in the input image.
 * 
 * @param src The `src` parameter is the input image that contains the food segment. It is of type
 * `cv::Mat`, which is a matrix data structure in OpenCV that represents an image.
 * @param mask The "mask" parameter is a binary image that represents the regions of interest in the
 * "src" image. It is used to segment the food objects in the image.
 */
void FoodSegmenter::refinePilawRice(const cv::Mat& src, cv::Mat& mask) {
    cv::Mat workingFood;
    cv::bitwise_and(src, src, workingFood, mask);
    // Split the image into individual BGR channels
    cv::Mat hsvImage, hslImage;
    cv::cvtColor(workingFood, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsvImage, channels);

    cv::Mat thresholdedMaskHSV = channels[1] > 0.35*255 ;
    cv::Mat sThresholded;
    cv::bitwise_and(workingFood, workingFood, sThresholded, thresholdedMaskHSV);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholdedMaskHSV, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(thresholdedMaskHSV.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i++) {
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

/**
 * The function `refineTomatoPasta` refines a given mask by performing various image processing
 * operations to isolate the tomato pasta in the source image.
 * 
 * @param src The source image on which the segmentation is performed. It is of type cv::Mat, which is
 * a matrix data structure in OpenCV representing an image.
 * @param mask The "mask" parameter is a reference to a cv::Mat object. It is used to store the
 * resulting mask after refining the tomato pasta image. The mask is a binary image where white pixels
 * represent the regions of interest (tomato pasta) and black pixels represent the background.
 */
void FoodSegmenter::refineTomatoPasta(const cv::Mat& src, cv::Mat& mask) {
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

    // cv::imshow("thresholdedRedChannel", thresholdedRedChannel);

    int kernelSizeClosing = 3;
    cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosing, kernelSizeClosing));
    // Perform dilation followed by erosion (closing operation)
    cv::Mat closingMask;
    cv::morphologyEx(thresholdedRedChannel, closingMask, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 5);
    cv::Mat closingImage;
    cv::bitwise_and(workingFood, workingFood, closingImage, closingMask);

    // cv::imshow("closingMask", closingMask);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closingMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(closingMask.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestAreaPost) {
            largestAreaPost = area;
            index = i;
        }
    }

    if (index != -1)
        cv::fillPoly(filledMask, contours[index], cv::Scalar(255));

    // cv::imshow("filledMask", filledMask);
    // cv::imshow("bgrThresholded", bgrThresholded);

    // cv::waitKey();
    mask = filledMask;
}


/**
 * The function refines a mask of a pork cutlet by closing holes, finding contours, sorting them by
 * area, and keeping the top n contours.
 * 
 * @param src The source image from which the pork cutlet is segmented.
 * @param mask The `mask` parameter is a binary image that represents the segmentation mask of a pork
 * cutlet in the input image `src`. The mask is initially provided as an input and will be refined and
 * updated within the `refinePorkCutlet` function.
 * @return Nothing is being returned. The function is modifying the "mask" parameter passed by
 * reference.
 */
void FoodSegmenter::refinePorkCutlet(const cv::Mat& src, cv::Mat& mask) {

    // Close the holes borders
    int closingSize = cv::boundingRect(mask).height / 4;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

/**
 * The function `refineMask` takes an input image and a mask, and refines the mask based on the
 * specified label.
 * 
 * @param src The source image on which the mask is based.
 * @param mask The `mask` parameter is a `cv::Mat` object that represents a binary mask image. It is
 * used to segment the food in the input image `src`. The mask should have the same size as the input
 * image, where the pixels corresponding to the food region are set to 255 (
 * @param label The "label" parameter is an integer value that represents the type of food segment that
 * needs to be refined.
 */
void FoodSegmenter::refineMask(const cv::Mat& src, cv::Mat& mask, int label) {
    switch (label) {
        case 1:
            refinePestoPasta(src, mask);
            break;
        case 5:
            refinePilawRice(src, mask);
            break;
        case 6:
            refinePorkCutlet(src, mask);
            break;
        default:
            break;
    }
}