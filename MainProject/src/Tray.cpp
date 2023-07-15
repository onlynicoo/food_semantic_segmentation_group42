#include <fstream>
#include "../include/FeatureComparator.h"
#include "../include/FindFood.h"
#include "../include/SegmentFood.h"
#include "../include/Utils.h"
#include "../include/Tray.h"


std::string Tray::get_traysAfterNames() {
    return traysAfterNames;
}

cv::Mat GetTrainedFeatures(std::string labelFeaturesPath) {
    cv::Mat labelFeatures;

    // Read template images
    cv::FileStorage fs(labelFeaturesPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Failed to open label features file." << std::endl;
    }
    fs["labelFeatures"] >> labelFeatures;
    fs.release();
    return labelFeatures;
}

std::map<int, cv::Vec3b> InitColorMap() {
    std::map<int, cv::Vec3b> colors;
    colors[0] = cv::Vec3b{0, 0, 0};  // Black
    colors[1] = cv::Vec3b{0, 255, 124};  // Green
    colors[2] = cv::Vec3b{0, 0, 255};  // Red
    colors[3] = cv::Vec3b{255, 0, 0};  // Blue
    colors[4] = cv::Vec3b{0, 255, 255};  // Yellow
    colors[5] = cv::Vec3b{255, 255, 0};  // Cyan
    colors[6] = cv::Vec3b{255, 0, 255};  // Magenta
    colors[7] = cv::Vec3b{0, 165, 255};  // Orange
    colors[8] = cv::Vec3b{128, 0, 128};  // Purple
    colors[9] = cv::Vec3b{203, 192, 255};  // Pink
    colors[10] = cv::Vec3b{51, 102, 153};  // Brown
    colors[11] = cv::Vec3b{128, 128, 128};  // Gray
    colors[12] = cv::Vec3b{255, 255, 255};  // White
    colors[13] = cv::Vec3b{128, 128, 0};  // Olive

    return colors;
}

void InsertBoundingBox(cv::Mat src, std::string filePath) {

    std::ofstream file(filePath, std::ios::trunc); // Open the file in append mode

    for (int i = 1; i < 14; i++) {

        cv::Mat binaryMask = (src == i);   

        if (cv::countNonZero(binaryMask) == 0)
            continue;

        cv::Rect bbox = cv::boundingRect(binaryMask);
        file << "ID: " << i << "; [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]\n"; // Write the new line to the file

        /*
        // Create multiple bbox
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<cv::Rect> boundingBoxes;
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect boundingBox = cv::boundingRect(contours[i]);
            boundingBoxes.push_back(boundingBox);
        }

        if (file.is_open() && boundingBoxes.size() > 0) {
            for (int j = 0; j < boundingBoxes.size(); j++)
                file << "ID " << i << "; [" << boundingBoxes[j].x << ", " << boundingBoxes[j].y << ", " << boundingBoxes[j].width << ", " << boundingBoxes[j].height << "]\n"; // Write the new line to the file
        }
        */
    }
    
    file.close();

}

cv::Mat Tray::SegmentImage(const cv::Mat src, std::vector<int>& labelsFound, std::string filePath) {
    // it contains
    // image detection | image segmentation

    std::vector<int> firstPlatesLabel{1, 2, 3, 4, 5}, secondPlatesLabel{6, 7, 8, 9}, sideDishesLabels{10, 11};
    int saladLabel = 12;

    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentationMask(segmentationMaskSize, CV_8UC1, cv::Scalar(0));

    cv::Mat labels = GetTrainedFeatures(labelFeaturesPath);
    std::vector<cv::Vec3f> plates = FindFood::findPlates(src);
    plates.resize(std::min(2, (int) plates.size()));                          // Assume there are at most 2 plates

    std::vector<std::vector<FeatureComparator::LabelDistance>> platesLabelDistances;
    std::vector<int> platesLabels, allowedLabels;
    std::vector<cv::Mat> platesMasks;

    allowedLabels = Utils::getVectorUnion(firstPlatesLabel, Utils::getVectorUnion(secondPlatesLabel, sideDishesLabels));
    if (!labelsFound.empty())
        allowedLabels = Utils::getVectorIntersection(allowedLabels, labelsFound);

    for(int i = 0; i < plates.size(); i++) {
        
        cv::Point center;
        int radius;
        center.x = plates[i][0];
        center.y = plates[i][1];
        radius = plates[i][2];

        cv::Mat tmpMask;
        
        SegmentFood::getFoodMask(src, tmpMask, center, radius);

        // Do not consider empty plate masks
        if (cv::countNonZero(tmpMask) == 0) {
            plates.erase(plates.begin() + i);
            i--;
            continue;
        }

        platesMasks.push_back(tmpMask);
        
        // Creates the features for the segmented patch
        cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, tmpMask);
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
            if (Utils::getIndexInVector(firstPlatesLabel, platesLabelDistances[0][j].label) != -1) {
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
            if (Utils::getIndexInVector(firstPlatesLabel, platesLabelDistances[1][j].label) != -1) {
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
        if (Utils::getIndexInVector(firstPlatesLabel, foodLabel) != -1) {
            
            platesLabels.push_back(foodLabel);

            // Refine segmentation
            SegmentFood::refineMask(src, platesMasks[i], foodLabel);
            // Add to segmentation mask
            for(int r = 0; r < segmentationMask.rows; r++)
                for(int c = 0; c < segmentationMask.cols; c++)
                    if(platesMasks[i].at<uchar>(r,c) != 0)
                        segmentationMask.at<uchar>(r,c) = int(platesMasks[i].at<uchar>(r,c)*foodLabel);

            std::cout << "Label found: " << LABELS[foodLabel] << std::endl;
        } else {

            // We have to split the mask into more foods
            allowedLabels = Utils::getVectorUnion(secondPlatesLabel, sideDishesLabels);  
            if (!labelsFound.empty())
                allowedLabels = Utils::getVectorIntersection(labelsFound, allowedLabels);

            // Crop mask to non zero area
            cv::Mat tmpMask = platesMasks[i];
            cv::Rect bbox = cv::boundingRect(tmpMask);
            cv::Mat croppedMask = tmpMask(bbox).clone();

            // Split mask into a grid of smaller masks
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
                    cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, curMask);
                    foodLabel = FeatureComparator::getLabelDistances(labels, allowedLabels, patchFeatures)[0].label;
                    gridLabels.push_back(foodLabel);
                }
            }

            // Repeat using only the most frequent labels of the previous step
            std::vector<int> sortedGridLabels = Utils::sortVectorByFreq(gridLabels);
            std::vector<int> foundSecondPlates = Utils::getVectorIntersection(sortedGridLabels, secondPlatesLabel);
            std::vector<int> foundSideDishes = Utils::getVectorIntersection(sortedGridLabels, sideDishesLabels);

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
                    cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, curMask);
                    foodLabel = FeatureComparator::getLabelDistances(labels, keptLabels, patchFeatures)[0].label;
                    labelMat[row][col] = foodLabel;
                }
            }

            // Re-assign labels based also on the neighboring submasks
            labelMat = Utils::getMostFrequentMatrix(labelMat, 1);

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
                platesLabels.push_back(foodLabel);

                SegmentFood::refineMask(src, platesMasks[i], foodLabel);

                for(int r = 0; r < segmentationMask.rows; r++)
                        for(int c = 0; c < segmentationMask.cols; c++)
                            if(foodMasks[j].at<uchar>(r,c) != 0)
                                segmentationMask.at<uchar>(r,c) = int(foodMasks[j].at<uchar>(r,c) * foodLabel);
                
                std::cout << "Label found: " << LABELS[foodLabel] << std::endl;
            }
        }
    }

    // Segment bread
    cv::Mat breadMask = SegmentFood::SegmentBread(src);
    for(int r = 0; r < breadMask.rows; r++)
        for(int c = 0; c < breadMask.cols; c++)
            if(segmentationMask.at<uchar>(r,c) == 0)
                segmentationMask.at<uchar>(r,c) = breadMask.at<uchar>(r,c);

    // Segment salad
    std::vector<cv::Vec3f> saladBowls = FindFood::findSaladBowl(src, (Utils::getIndexInVector(labelsFound, saladLabel) != -1));
    if (saladBowls.size() != 0) {

        cv::Point center;
        int radius;
        center.x = saladBowls[0][0];
        center.y = saladBowls[0][1];
        radius = saladBowls[0][2];

        cv::Mat saladMask;
        SegmentFood::getSaladMask(src, saladMask, center, radius);

        for(int r = 0; r < saladMask.rows; r++)
            for(int c = 0; c < saladMask.cols; c++)
                if(segmentationMask.at<uchar>(r,c) == 0)
                    segmentationMask.at<uchar>(r,c) = int(saladMask.at<uchar>(r,c) * saladLabel);

        platesLabels.push_back(saladLabel);
    }
    
    // Keep labels found
    labelsFound = platesLabels;

    return segmentationMask;
}

std::string ExtractTray(std::string imagePath) {
    
    // Find the penultimate occurrence of '/'
    size_t lastSlashPos = imagePath.rfind('/');
    size_t penultimateSlashPos = imagePath.rfind('/', lastSlashPos - 1);

    // Extract the substring between the penultimate and last slash
    std::string penultimatePath = imagePath.substr(penultimateSlashPos + 1, lastSlashPos - penultimateSlashPos - 1);

    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return "../output/" + penultimatePath;
}

std::string ExtractName(std::string imagePath) {

    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return  imageName.append("_bounding_box");
}

//should make bool and check output
void Tray::SaveSegmentedMask(std::string path, cv::Mat src) {

    std::string imageName = ExtractName(path);

    if (imageName.compare("food_image_bounding_box") == 0)
        imageName = "food_image_mask";
    if (imageName.compare("leftover1_bounding_box") == 0)
        imageName = "leftover1";
    if (imageName.compare("leftover2_bounding_box") == 0)
        imageName = "leftover2";
    if (imageName.compare("leftover3_bounding_box") == 0)
        imageName = "leftover3";

    std::string filename = ExtractTray(path) + "/" + "masks/" + imageName + ".png";

    bool success = cv::imwrite(filename, src);

}

// constructor that does everything
Tray::Tray(std::string trayBefore, std::string trayAfter) {

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;

    traysBefore = before;
    traysAfter = after;

    std::vector<int> labelsFound;
    traysBeforeDetected = ExtractTray(trayBefore) + "/" + "bounding_boxes/" + ExtractName(trayBefore) + ".txt";
    traysAfterDetected = ExtractTray(trayAfter) + "/" + "bounding_boxes/" + ExtractName(trayAfter) + ".txt";

    traysBeforeSegmented = SegmentImage(before, labelsFound, traysBeforeDetected);
    traysAfterSegmented = SegmentImage(after, labelsFound,  traysAfterDetected);

    InsertBoundingBox(traysBeforeSegmented, traysBeforeDetected);
    InsertBoundingBox(traysAfterSegmented, traysAfterDetected);

    SaveSegmentedMask(traysBeforeNames, traysBeforeSegmented);
    SaveSegmentedMask(traysAfterNames, traysAfterSegmented);
    
}

cv::Mat SegmentedImageFromMask(cv::Mat src) {
    std::map<int, cv::Vec3b> colors = InitColorMap();
    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentedImage(segmentationMaskSize, CV_8UC3, cv::Scalar(0));
    for(int r = 0; r < src.rows; r++)
        for(int c = 0; c < src.cols; c++)
            if(src.at<uchar>(r,c) != 0)
                segmentedImage.at<cv::Vec3b>(r,c) = colors[int(src.at<uchar>(r,c))];
    return segmentedImage;
}

cv::Mat OverimposeDetection(cv::Mat src, std::string filePath) {

    cv::Mat out = src.clone();
    std::map<int, cv::Vec3b> colors = InitColorMap();

    std::ifstream file(filePath); 
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {

            cv::Point topLeft, bottomRight;

            // Extracting 'x'
            size_t xIndex = line.find("ID") + 3;
            size_t semicolonIndex = line.find(";", xIndex);
            int foodId = std::stoi(line.substr(xIndex, semicolonIndex - xIndex));

            // Extracting 'a', 'b', 'c', 'd'
            size_t openBracketIndex = line.find("[", semicolonIndex) + 1;
            size_t closeBracketIndex = line.find("]", openBracketIndex);
            std::string elementsStr = line.substr(openBracketIndex, closeBracketIndex - openBracketIndex);

            std::vector<int> elements;
            size_t commaIndex = 0;
            size_t nextCommaIndex = elementsStr.find(",", commaIndex);
            while (nextCommaIndex != std::string::npos) {
                elements.push_back(std::stoi(elementsStr.substr(commaIndex, nextCommaIndex - commaIndex)));
                commaIndex = nextCommaIndex + 2; // Skip comma and space
                nextCommaIndex = elementsStr.find(",", commaIndex);
            }
            elements.push_back(std::stoi(elementsStr.substr(commaIndex))); // Last element

            // Assigning values to points variables
            if (elements.size() == 4) {
                topLeft.x = elements[0];
                topLeft.y = elements[1];
                bottomRight.x = elements[0] + elements[2];
                bottomRight.y = elements[1] + elements[3];
            }

            cv::rectangle(out, topLeft, bottomRight, colors[foodId], 10);  // Draw the rectangle on the image
        }
    }
    return out;
}

void Tray::PrintInfo() {

    std::string window_name = "info tray";

    cv::Mat imageGrid, imageRow;
    cv::Size stdSize(0,0);
    stdSize = traysBefore.size();

    cv::Mat tmp1_1, tmp1_2, tmp1_3, tmp2_1, tmp2_2, tmp2_3; 

    cv::Mat colorBeforeSegmented = SegmentedImageFromMask(traysBeforeSegmented);
    cv::Mat colorAfterSegmented = SegmentedImageFromMask(traysAfterSegmented);

    tmp1_1 = traysBefore.clone();

    // Resize output to have all images of same size
    resize(traysAfter, tmp2_1, stdSize);
    // resize(OverimposeDetection(traysBefore, traysBeforeDetected), tmp1_2, stdSize);
    // resize(OverimposeDetection(traysAfter, traysAfterDetected), tmp2_2, stdSize);

    std::vector<cv::Vec3f> saladBefore = FindFood::findSaladBowl(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
        saladAfter = FindFood::findSaladBowl(traysAfter, false);
    if (saladBefore.size() == 0) {
    }   
    else {
        saladAfter = FindFood::findSaladBowl(traysAfter, true);
    }

    resize(OverimposeDetection(traysBefore, traysBeforeDetected), tmp1_2, stdSize);
    resize(OverimposeDetection(traysAfter, traysAfterDetected), tmp2_2, stdSize);
    resize(colorBeforeSegmented, tmp1_3, stdSize);
    resize(colorAfterSegmented, tmp2_3, stdSize);

    // Add image to current image row
    tmp2_3.copyTo(imageRow);
    hconcat(tmp2_2, imageRow, imageRow);
    hconcat(tmp2_1, imageRow, imageRow);
    imageRow.copyTo(imageGrid);
    imageRow.release();

    tmp1_3.copyTo(imageRow);
    hconcat(tmp1_2, imageRow, imageRow);
    hconcat(tmp1_1, imageRow, imageRow);
    vconcat(imageRow, imageGrid, imageGrid);
    imageRow.release();

    // Resize the full image grid and display it
    resize(imageGrid, imageGrid, stdSize);
    imshow(window_name, imageGrid);

    cv::waitKey(0);
}


void Tray::PrintSaladPlate() {

    std::vector<cv::Vec3f> saladBefore = FindFood::findSaladBowl(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
    if (saladBefore.size() == 0)
        saladAfter = FindFood::findSaladBowl(traysAfter, false);
    else 
        saladAfter = FindFood::findSaladBowl(traysAfter, true);

    cv::imshow("before", FindFood::drawPlates(traysBefore, saladBefore)); 
    cv::imshow("after", FindFood::drawPlates(traysAfter, saladAfter)); 
    
    cv::waitKey();

}