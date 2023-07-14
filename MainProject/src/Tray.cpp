#include "../include/Tray.h"
#include <iostream>
#include <fstream>

//
std::string Tray::get_traysAfterNames() {
    return traysAfterNames;
}        

cv::Mat Tray::SegmentFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
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

std::vector<int> sortVectorByFreq(std::vector<int> values) {
    // Create frequency map
    std::map<int, int> freqMap;
    for (int value : values)
        freqMap[value]++;

    // Sort values descending based on frequencies
    std::vector<std::pair<int, int>> sortedFreq(freqMap.begin(), freqMap.end());
    std::sort(sortedFreq.begin(), sortedFreq.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
        return a.second > b.second;
    });

    // Get n most frequent values
    std::vector<int> result;
    for (int i = 0; i < sortedFreq.size(); i++)
        result.push_back(sortedFreq[i].first);
    return result;
}

int getMostFrequentNeighbor(std::vector<std::vector<int>> matrix, int row, int col, int radius) {
    std::unordered_map<int, int> freqMap;
    int maxCount = -1, mostFreqValue = -1;

    // Iterate over the neighbors
    for (int i = row - radius; i <= row + radius; i++) {
        for (int j = col - radius; j <= col + radius; j++) {
            
            if (i < 0 || j < 0 || i >= matrix.size() || j >= matrix[0].size() || (i == row && j == col) || matrix[i][j] == -1)
                continue;

            int neighborValue = matrix[i][j];
            freqMap[neighborValue]++;
            int count = freqMap[neighborValue];

            if (count > maxCount) {
                maxCount = count;
                mostFreqValue = neighborValue;
            }
        }
    }
    return mostFreqValue;
}

std::vector<std::vector<int>> getMostFrequentMatrix(std::vector<std::vector<int>> matrix, int radius) {
    std::vector<std::vector<int>> newMatrix(matrix.size(), std::vector<int>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); i++)
        for (int j = 0; j < matrix[0].size(); j++)
                newMatrix[i][j] = getMostFrequentNeighbor(matrix, i, j, radius);

    return newMatrix;
}

std::vector<int> getVectorUnion(std::vector<int> a, std::vector<int> b){
    std::vector<int> c;
    c.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(c));
    std::copy(b.begin(), b.end(), std::back_inserter(c));
    return c;
}

std::vector<int> getVectorIntersection(std::vector<int> a, std::vector<int> b){
    std::set<int> setB(b.begin(), b.end());
    std::vector<int> c;
    for (int i = 0; i < a.size(); i++)
        if (setB.count(a[i]) > 0)
            c.push_back(a[i]);
    return c;
}

int getIndexInVector(std::vector<int> vector, int value) {
    auto it = std::find(vector.begin(), vector.end(), value);
    if (it != vector.end())
        return std::distance(vector.begin(), it);
    return -1;
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
    std::string LABELS[14] = {
        "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
        "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
        "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

    std::string labelFeaturesPath = "../features/label_features.yml";

    std::vector<int> firstPlatesLabel{1, 2, 3, 4, 5}, secondPlatesLabel{6, 7, 8, 9}, sideDishesLabels{10, 11};
    int saladLabel = 12;

    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentationMask(segmentationMaskSize, CV_8UC1, cv::Scalar(0));

    cv::Mat labels = GetTrainedFeatures(labelFeaturesPath);
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    plates.resize(std::min(2, (int) plates.size()));                          // Assume there are at most 2 plates

    std::vector<std::vector<FeatureComparator::LabelDistance>> platesLabelDistances;
    std::vector<int> platesLabels, allowedLabels;
    std::vector<cv::Mat> platesMasks;

    allowedLabels = getVectorUnion(firstPlatesLabel, getVectorUnion(secondPlatesLabel, sideDishesLabels));
    if (!labelsFound.empty())
        allowedLabels = getVectorIntersection(allowedLabels, labelsFound);

    for(int i = 0; i < plates.size(); i++) {
        
        cv::Point center;
        int radius;
        center.x = plates[i][0];
        center.y = plates[i][1];
        radius = plates[i][2];

        cv::Mat tmpMask;
        
        PlateRemover::getFoodMask(src, tmpMask, center, radius);

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
            if (getIndexInVector(firstPlatesLabel, platesLabelDistances[0][j].label) != -1) {
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
            if (getIndexInVector(firstPlatesLabel, platesLabelDistances[1][j].label) != -1) {
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
        if (getIndexInVector(firstPlatesLabel, foodLabel) != -1) {
            
            platesLabels.push_back(foodLabel);

            // Add to segmentation mask
            for(int r = 0; r < segmentationMask.rows; r++)
                for(int c = 0; c < segmentationMask.cols; c++)
                    if(platesMasks[i].at<uchar>(r,c) != 0)
                        segmentationMask.at<uchar>(r,c) = int(platesMasks[i].at<uchar>(r,c)*foodLabel);

            std::cout << "Label found: " << LABELS[foodLabel] << std::endl;
        } else {

            // We have to split the mask into more foods
            allowedLabels = getVectorUnion(secondPlatesLabel, sideDishesLabels);  
            if (!labelsFound.empty())
                allowedLabels = getVectorIntersection(labelsFound, allowedLabels);

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
            std::vector<int> sortedGridLabels = sortVectorByFreq(gridLabels);
            std::vector<int> foundSecondPlates = getVectorIntersection(sortedGridLabels, secondPlatesLabel);
            std::vector<int> foundSideDishes = getVectorIntersection(sortedGridLabels, sideDishesLabels);

            std::vector<int> keptLabels;
            if (!foundSecondPlates.empty())
                keptLabels.push_back(foundSecondPlates[0]);
            if (!foundSideDishes.empty())
                keptLabels.push_back(foundSideDishes[0]);
            if (foundSideDishes.size() > 1)
                keptLabels.push_back(foundSideDishes[1]);

            std::cout << "Kept labels: ";
            for (int j = 0; j < keptLabels.size(); j++)
                std::cout << keptLabels[j] << " ";
            std::cout << std::endl;

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
            labelMat = getMostFrequentMatrix(labelMat, 1);

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
                    int index = getIndexInVector(keptLabels, foodLabel);

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

                for(int r = 0; r < segmentationMask.rows; r++)
                        for(int c = 0; c < segmentationMask.cols; c++)
                            if(foodMasks[j].at<uchar>(r,c) != 0)
                                segmentationMask.at<uchar>(r,c) = int(foodMasks[j].at<uchar>(r,c) * foodLabel);
                
                std::cout << "Label found: " << LABELS[foodLabel] << std::endl;
            }
        }
    }

    // Segment bread
    cv::Mat breadMask = SegmentBread(src);
    for(int r = 0; r < breadMask.rows; r++)
        for(int c = 0; c < breadMask.cols; c++)
            if(segmentationMask.at<uchar>(r,c) == 0)
                segmentationMask.at<uchar>(r,c) = breadMask.at<uchar>(r,c);

    // Segment salad
    std::vector<cv::Vec3f> saladBowls = PlatesFinder::get_salad(src, (getIndexInVector(labelsFound, saladLabel) != -1));
    if (saladBowls.size() != 0) {

        cv::Point center;
        int radius;
        center.x = saladBowls[0][0];
        center.y = saladBowls[0][1];
        radius = saladBowls[0][2];

        cv::Mat saladMask;
        PlateRemover::getSaladMask(src, saladMask, center, radius);

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

cv::Mat Tray::SegmentBread(cv::Mat src) {

    // used as base img
    cv::Mat maskedImage = src.clone();

    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    std::vector<cv::Vec3f> salad = PlatesFinder::get_salad(src, true);

    // Draw the circle
    for( size_t i = 0; i < plates.size(); i++ ) {
        cv::Vec3i c = plates[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle outline
        int radius = c[2];
        circle(maskedImage, center, radius*1, cv::Scalar(0,0,0), cv::FILLED);
    }

    // Draw the circle
    for( size_t i = 0; i < salad.size(); i++ ) {
        cv::Vec3i c = salad[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle outline
        int radius = c[2];
        circle(maskedImage, center, radius*1.4, cv::Scalar(0,0,0), cv::FILLED);
    }

    // Convert image to YUV color space
    cv::Mat yuvImage;
    cv::cvtColor(maskedImage, yuvImage, cv::COLOR_BGR2YUV);

    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> yuvChannels;
    cv::split(yuvImage, yuvChannels);

    // Create a binary mask of pixels within the specified range
    cv::Mat mask;
    // Put this in .h file
    int thresholdLow = 70;
    int thresholdHigh = 117;
    cv::inRange(yuvChannels[1], thresholdLow, thresholdHigh, mask);

    // Apply the mask to the original image
    cv::Mat resultyuv;
    cv::bitwise_and(yuvChannels[1], yuvChannels[1], resultyuv, mask);


    // Define the structuring element for morphological operation
    cv::Mat kernelclosure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

    // Perform morphological closure
    cv::Mat resultclosure;
    cv::morphologyEx(resultyuv, resultclosure, cv::MORPH_CLOSE, kernelclosure);
    

    // Perform connected component analysis
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(resultclosure, labels, stats, centroids);

    // Find the connected component with the largest area
    int largestComponent = 0;
    int largestArea = 0;
    for (int i = 1; i < numComponents; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > largestArea) {
            largestArea = area;
            largestComponent = i;
        }
    }

    // Create a binary mask for the largest component
    cv::Mat largestComponentMask = (labels == largestComponent);

    cv::Mat kernel = (cv::Mat_<float>(51, 51, 1))/(45*45);
    
    // Apply the sliding kernel using filter2D
    cv::Mat result;
    cv::filter2D(largestComponentMask, result, -1, kernel);

    // Threshold the result image
    cv::Mat thresholdedLargestComponentMask;
    double maxValue = 255;

    // put this in .h file
    int thresholdValueToChange = 111;
    cv::threshold(result, thresholdedLargestComponentMask, thresholdValueToChange, maxValue, cv::THRESH_BINARY);



    
    // Apply the mask to the original image
    cv::Mat resultlargestComponents;
    cv::bitwise_and(src, src, resultlargestComponents, thresholdedLargestComponentMask);


    // Apply Canny edge detection
    cv::Mat gray;
    cv::cvtColor(resultlargestComponents, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    int t1 = 50, t2 = 150;
    cv::Canny(gray, edges, t1, t2);


    cv::Mat kernelCanny = (cv::Mat_<float>(7, 7, 1))/(7*7);
    
    // Apply the sliding kernel using filter2D
    cv::Mat resultCanny;
    cv::filter2D(edges, result, -1, kernelCanny);

    // Threshold the result image
    cv::Mat thresholdedCanny;
    double maxValueCanny = 255;

    // put this in .h file
    int thresholdValueCanny = 115;
    cv::threshold(result, thresholdedCanny, thresholdValueCanny, maxValue, cv::THRESH_BINARY);
    
    // Define the structuring element for closing operation
    int kernelSizeDilation = 3; // Adjust the size according to your needs
    int kernelSizeClosing = 5; // Adjust the size according to your needs
    int kernelSizeErosion = 3; // Adjust the size according to your needs
    cv::Mat kernelDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeDilation, kernelSizeDilation));
    cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeClosing, kernelSizeClosing));
    cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeErosion, kernelSizeErosion));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat closedImage;
    cv::morphologyEx(thresholdedCanny, closedImage, cv::MORPH_DILATE, kernelDilation, cv::Point(1, 1), 10);
    cv::morphologyEx(closedImage, closedImage, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 4);

    


    cv::morphologyEx(closedImage, closedImage, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 6);

    cv::morphologyEx(closedImage, closedImage, cv::MORPH_DILATE, kernelDilation, cv::Point(1, 1), 20);


    // Apply the mask to the original image
    cv::Mat res;
    cv::bitwise_and(src, src, res, closedImage);


    //not bad but to improve
    // Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(res, hsvImage, cv::COLOR_BGR2HSV);

        // Split HSV channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Access and process the H, S, and V channels separately
    cv::Mat hueChannel = hsvChannels[0];
    cv::Mat saturationChannel = hsvChannels[1];
    cv::Mat valueChannel = hsvChannels[2];


    // Threshold the result image
    cv::Mat thresholdedSaturation;
    cv::Mat saturationMask;
    //put this in .h file
    int thresholdSaturation = 140;
    cv::threshold(saturationChannel, thresholdedSaturation, thresholdSaturation, 255 , cv::THRESH_BINARY);
    cv::threshold(saturationChannel, saturationMask, 1, 255 , cv::THRESH_BINARY);

    cv::Mat newMask = saturationMask - thresholdedSaturation;

    cv::morphologyEx(newMask, newMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 1);
    cv::morphologyEx(newMask, newMask, cv::MORPH_DILATE, kernelErosion, cv::Point(1, 1), 12);
        


    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(newMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(newMask.size(), CV_8UC1);

    double thresholdArea = 20000;
    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
        if (area > thresholdArea)
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }
    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(13));



    cv::Mat out = filledMask;
    

    
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

    std::vector<cv::Vec3f> saladBefore = PlatesFinder::get_salad(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
        saladAfter = PlatesFinder::get_salad(traysAfter, false);
    if (saladBefore.size() == 0) {
    }   
    else {
        saladAfter = PlatesFinder::get_salad(traysAfter, true);
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

    std::vector<cv::Vec3f> saladBefore = PlatesFinder::get_salad(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
    if (saladBefore.size() == 0)
        saladAfter = PlatesFinder::get_salad(traysAfter, false);
    else 
        saladAfter = PlatesFinder::get_salad(traysAfter, true);

    cv::imshow("before", PlatesFinder::print_plates_image(traysBefore, saladBefore)); 
    cv::imshow("after", PlatesFinder::print_plates_image(traysAfter, saladAfter)); 
    
    cv::waitKey();

}

cv::Mat RefinePastaPesto(cv::Mat src, cv::Mat mask) {
    cv::Mat workingFood;
    cv::bitwise_and(src, src, workingFood, mask);
    
    //cv::imshow("tmp1", workingFood); cv::waitKey();

    cv::Mat hsvImage;
    cv::cvtColor(workingFood, hsvImage, cv::COLOR_BGR2HSV);
    
    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    //cv::imshow("channel0", hsvChannels[0]); 
    //cv::imshow("channel1", hsvChannels[1]); 
    //cv::imshow("channel2", hsvChannels[2]); 
    
    //cv::imshow("hsvImage", hsvImage);


    cv::Mat thresholdingMask = hsvChannels[1] > 0.6*255;
    cv::Mat thresholdedImage;
    cv::bitwise_and(src, src, thresholdedImage, thresholdingMask);

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

    //cv::imshow("thresholdedGreenMask", thresholdedGreenMask);

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

    cv::Mat maskedImg;
    cv::bitwise_and(src, src, maskedImg, filledMask);
    
    //cv::imshow("output", maskedImg);

    //cv::waitKey();


    return workingFood;
}

cv::Mat thresholdBGR(const cv::Mat& image, const cv::Scalar& lowerThreshold, const cv::Scalar& upperThreshold) {
    // Create a binary mask using BGR thresholds
    cv::Mat mask;
    cv::inRange(image, lowerThreshold, upperThreshold, mask);

    // Apply the mask to the original image
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    return mask;
}

cv::Mat RefinePastaTomato(cv::Mat src, cv::Mat mask) {
    cv::Mat workingFood;
    cv::bitwise_and(src, src, workingFood, mask);
    cv::imshow("workingFood", workingFood);  cv::waitKey();
    
    // Split the image into individual BGR channels
    std::vector<cv::Mat> channels;
    cv::split(workingFood, channels);

    // Keep only the red channel
    cv::Mat redChannel = channels[2];



    cv::Mat thresholdedRedChannel = redChannel > 160;
    cv::Mat bgrThresholded;
    cv::bitwise_and(workingFood, workingFood, bgrThresholded, thresholdedRedChannel);
    
    cv::imshow("thresholdedRedChannel", thresholdedRedChannel);


    int kernelSizeClosing = 3;
    cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosing, kernelSizeClosing));
    // Perform dilation followed by erosion (closing operation)
    cv::Mat closingMask;
    cv::morphologyEx(thresholdedRedChannel, closingMask, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 5);
    cv::Mat closingImage;
    cv::bitwise_and(workingFood, workingFood, closingImage, closingMask);


    cv::imshow("closingMask", closingMask);





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


    cv::imshow("filledMask", filledMask);
    
    cv::imshow("bgrThresholded", bgrThresholded);

    cv::waitKey();
    return bgrThresholded ;
}

// test to refine segmentations of leftover
void Tray::RefineSegmentation() {

    for (int i = 1; i < 14; i++) {
        cv::Mat givenFoodMatBefore = (traysBeforeSegmented == i);
        cv::Mat givenFoodMatAfter = (traysAfterSegmented == i);
        switch (i) {
            case 1:
                RefinePastaPesto(traysBefore, givenFoodMatBefore);
                RefinePastaPesto(traysAfter, givenFoodMatAfter);
                break;
            case 2:
                RefinePastaTomato(traysBefore, givenFoodMatBefore);
                RefinePastaTomato(traysAfter, givenFoodMatAfter);
                break;
            default:
                break;
        }
    }
}