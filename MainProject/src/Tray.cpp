#include "../include/Tray.h"
#include <iostream>
#include <fstream>

cv::Mat Tray::DetectFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
}

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
            
            if (i < 0 || j < 0 || i >= matrix.size() || j >= matrix[0].size())
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
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<int> c;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), back_inserter(c));
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

    std::string labelFeaturesPath = "../data/label_features.yml";

    std::vector<int> firstPlatesLabel{1, 2, 3, 4, 5}, secondPlatesLabel{6, 7, 8, 9}, sideDishesLabels{10, 11};

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

    /*for (int j = 0; j < platesLabelDistances.size(); j++) {
        for (int k = 0; k < platesLabelDistances[j].size(); k++) {
            std::cout << platesLabelDistances[j][k].label << " " << platesLabelDistances[j][k].distance << " - ";
        }
        std::cout << std::endl;
    }*/

    // Choose best labels such that if there are more plates, they are one first and one second plate
    if (platesLabelDistances.size() > 1) {

        while (!((getIndexInVector(firstPlatesLabel, platesLabelDistances[0][0].label) != -1)
                ^ (getIndexInVector(firstPlatesLabel, platesLabelDistances[1][0].label) != -1)))
        {
            if (platesLabelDistances[0][0] < platesLabelDistances[1][0])
                platesLabelDistances[1].erase(platesLabelDistances[1].begin());
            else
                platesLabelDistances[0].erase(platesLabelDistances[0].begin());
        }
    }

    for (int i = 0; i < plates.size(); i++) {
        std::cout << "Plate " << i << std::endl;

        int foodLabel = platesLabelDistances[i][0].label;
        double labelDistance = platesLabelDistances[i][0].distance;
        
        // If it is a first plate or if we have enough confidence with the label, we have finished
        if (getIndexInVector(firstPlatesLabel, foodLabel) != -1 || labelDistance < FeatureComparator::NORMALIZE_VALUE * 0.07) {
            
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

            // Add found masks to the segmentation mask
            for (int j = 0; j < keptLabels.size(); j++) {

                if (cv::countNonZero(foodMasks[j]) == 0)
                        continue;

                foodLabel = keptLabels[j];
                platesLabels.push_back(foodLabel);

                for(int r = 0; r < segmentationMask.rows; r++)
                        for(int c = 0; c < segmentationMask.cols; c++)
                            if(foodMasks[j].at<uchar>(r,c) != 0)
                                segmentationMask.at<uchar>(r,c) = int(foodMasks[j].at<uchar>(r,c)*foodLabel);
                
                std::cout << "Label found: " << LABELS[foodLabel] << std::endl;
            }
        }
    }

    // Keep labels found
    labelsFound = platesLabels;

    return segmentationMask;
}

std::string ExtractName(std::string imagePath) {
    
    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return "../output/" + imageName + ".txt";
}

Tray::Tray(std::string trayBefore, std::string trayAfter) {

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;

    traysBefore = before;
    traysAfter = after;

    std::vector<int> labelsFound;
    traysBeforeDetected = ExtractName(trayBefore);
    traysAfterDetected = ExtractName(trayAfter);

    traysBeforeSegmented = SegmentImage(before, labelsFound, traysBeforeDetected);
    traysAfterSegmented = SegmentImage(after, labelsFound,  traysAfterDetected);

    InsertBoundingBox(traysBeforeSegmented, traysBeforeDetected);
    InsertBoundingBox(traysAfterSegmented, traysAfterDetected);
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

    std::vector<cv::Vec3f> saladBefore = PlatesFinder::get_salad(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
        saladAfter = PlatesFinder::get_salad(traysAfter, false);
    if (saladBefore.size() == 0) {
    }   
    else {
        saladAfter = PlatesFinder::get_salad(traysAfter, true);
    }

    resize(PlatesFinder::print_plates_image(OverimposeDetection(traysBefore, traysBeforeDetected), saladBefore), tmp1_2, stdSize);
    resize(PlatesFinder::print_plates_image(OverimposeDetection(traysAfter, traysAfterDetected), saladAfter), tmp2_2, stdSize);
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