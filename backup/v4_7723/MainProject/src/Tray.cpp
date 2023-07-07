#include "../include/Tray.h"
#include <iostream>
#include <fstream>

cv::Mat Tray::DetectFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
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
    colors[10] = cv::Vec3b{42, 42, 165};  // Brown
    colors[11] = cv::Vec3b{128, 128, 128};  // Gray
    colors[12] = cv::Vec3b{255, 255, 255};  // White
    colors[13] = cv::Vec3b{128, 128, 0};  // Olive

    return colors;
}

std::vector<int> findThreeMostFrequent(const std::vector<int>& values) {
    // Create a frequency map to count the occurrences of each value
    std::map<int, int> frequencyMap;
    for (int value : values) {
        frequencyMap[value]++;
    }

    // Sort the values based on their frequencies in descending order
    std::vector<std::pair<int, int>> sortedFreq(frequencyMap.begin(), frequencyMap.end());
    std::sort(sortedFreq.begin(), sortedFreq.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second > b.second;
              });

    // Extract the three most frequent values
    std::vector<int> result;
    for (int i = 0; i < std::min(3, static_cast<int>(sortedFreq.size())); i++) {
        result.push_back(sortedFreq[i].first);
    }

    return result;
}

// Function to get the most frequent value among neighbors
int getMostFrequentNeighbor(const std::vector<std::vector<int>>& arr, int row, int col) {
    std::unordered_map<int, int> freqMap;
    int maxCount = 0;
    int mostFrequentValue = 0;

    // Iterate over the 8 neighboring cells
    for (int i = row - 1; i <= row + 1; ++i) {
        for (int j = col - 1; j <= col + 1; ++j) {
            // Skip if the indices are out of bounds or the current cell itself
            if (i < 0 || j < 0 || i >= arr.size() || j >= arr[0].size() || (i == row && j == col))
                continue;

            int neighborValue = arr[i][j];
            freqMap[neighborValue]++;
            int count = freqMap[neighborValue];

            // Update the most frequent value if necessary
            if (count > maxCount) {
                maxCount = count;
                mostFrequentValue = neighborValue;
            }
        }
    }

    return mostFrequentValue;
}

// Function to create a new array with most frequent neighbors
std::vector<std::vector<int>> createNewArray(const std::vector<std::vector<int>>& arr) {
    std::vector<std::vector<int>> newArray(arr.size(), std::vector<int>(arr[0].size()));

    // Iterate over each cell of the original array
    for (int i = 0; i < arr.size(); ++i) {
        for (int j = 0; j < arr[0].size(); ++j) {
            // Get the most frequent neighbor value for the current cell
            int mostFrequentNeighbor = getMostFrequentNeighbor(arr, i, j);
            newArray[i][j] = mostFrequentNeighbor;
        }
    }

    return newArray;
}

void InsertBoundingBox(cv::Mat src, int foodId, std::string filePath, std::ofstream& file) {
    cv::Mat binaryMask;
    cv::threshold(src, binaryMask, 0, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boundingBoxes;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect boundingBox = cv::boundingRect(contours[i]);
        boundingBoxes.push_back(boundingBox);
    }

    if (file.is_open() && boundingBoxes.size() > 0) {
        file << "ID " << foodId << "; [" << boundingBoxes[0].x << ", " << boundingBoxes[0].y << ", " << boundingBoxes[0].width << ", " << boundingBoxes[0].height << "]\n"; // Write the new line to the file
    }
}

cv::Mat Tray::SegmentImage(const cv::Mat src, std::vector<int>& labelsFound, std::string filePath) {
    // it contains
    // image detection | image segmentation
    std::string LABELS[14] = {
        "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
        "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
        "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

    std::ofstream file(filePath, std::ios::trunc); // Open the file in append mode


    std::string labelFeaturesPath = "../data/label_features.yml";

    std::vector<int> firstPlatesLabel{1, 2, 3, 4, 5};
    std::vector<int> secondPlatesLabel{6, 7, 8, 9, 10, 11};

    std::vector<int> labelWhitelist;
    if(labelsFound.size() == 0) {

        labelWhitelist.reserve(firstPlatesLabel.size() + secondPlatesLabel.size());
        std::copy(firstPlatesLabel.begin(), firstPlatesLabel.end(), std::back_inserter(labelWhitelist));
        std::copy(secondPlatesLabel.begin(), secondPlatesLabel.end(), std::back_inserter(labelWhitelist));
    } else {
        labelWhitelist = std::vector<int>(labelsFound);
    }
    

    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentationMask(segmentationMaskSize, CV_8UC1, cv::Scalar(0));

    cv::Mat labels = GetTrainedFeatures(labelFeaturesPath);
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    plates.resize(std::min(2, (int) plates.size()));                          // Assume there are at most 2 plates

    std::vector<std::vector<FeatureComparator::LabelDistance>> platesLabelDistances;
    std::vector<cv::Mat> platesMasks;

    for(int i = 0; i < plates.size(); i++) {
        
        cv::Point center;
        int radius;
        center.x = plates[i][0];
        center.y = plates[i][1];
        radius = plates[i][2];

        cv::Mat tmpMask;
        
        // Remove plates giving only food
        PlateRemover::getFoodMask(src, tmpMask, center, radius);
        platesMasks.push_back(tmpMask);
        
        // Creates the features for the segmented patch
        cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, tmpMask);

        platesLabelDistances.push_back(FeatureComparator::getLabelDistances(labels, labelWhitelist, patchFeatures));
    }

    // Choose best labels such that if there are more plates, they are one first and one second plate
    if (platesLabelDistances.size() > 1) {
        while (!(platesLabelDistances[0][0].label >= MIN_FIRST_PLATE_LABEL && platesLabelDistances[0][0].label <= MAX_FIRST_PLATE_LABEL
                ^ platesLabelDistances[1][0].label >= MIN_FIRST_PLATE_LABEL && platesLabelDistances[1][0].label <= MAX_FIRST_PLATE_LABEL))
        {
            if (platesLabelDistances[0][0] < platesLabelDistances[1][0])
                platesLabelDistances[1].erase(platesLabelDistances[1].begin());
            else
                platesLabelDistances[0].erase(platesLabelDistances[0].begin());
        }
    }

    for (int i = 0; i < plates.size(); i++) {

        int foodLabel = platesLabelDistances[i][0].label;
        std::cout << "Plate " << i << " label found: " << LABELS[foodLabel] << "\n";
        if(std::find(std::begin(firstPlatesLabel), std::end(firstPlatesLabel), foodLabel) != std::end(firstPlatesLabel)) {

            //updates the boundingbox file
            InsertBoundingBox(platesMasks[i], foodLabel, filePath, file);

            //create the segmentation image
            labelsFound.push_back(foodLabel);
            for(int r = 0; r < segmentationMask.rows; r++)
                for(int c = 0; c < segmentationMask.cols; c++)
                    if(platesMasks[i].at<uchar>(r,c) != 0)
                        segmentationMask.at<uchar>(r,c) = int(platesMasks[i].at<uchar>(r,c) * foodLabel);
        }
        else {
            //add refinition of segmentation mask for multifood plates
            labelsFound.push_back(foodLabel);

            std::cout << "Splitting food..." << std::endl;

            std::vector<int> mostFreq;

            cv::Mat tmpMask = platesMasks[i];
            cv::Rect bbox = cv::boundingRect(tmpMask);
            cv::Mat croppedMask = tmpMask(bbox).clone();
            int gridSize = 10;
            int windowSize = croppedMask.rows / gridSize;

            //std::cout << "second plates" << std::endl;

            for (int y = 0; y < croppedMask.rows - windowSize; y += windowSize) {
                for (int x = 0; x < croppedMask.cols - windowSize; x += windowSize) {

                    cv::Rect windowRect(x, y, windowSize, windowSize);
                    cv::Mat submask = croppedMask(windowRect).clone();

                    cv::Mat otherMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(otherMask(bbox)(windowRect));

                    /*cv::Mat masked;
                    cv::bitwise_and(src, src, masked, otherMask);
                    cv::imshow("masked", masked);cv::waitKey();*/

                    cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, otherMask);
                    foodLabel = FeatureComparator::getLabelDistances(labels, secondPlatesLabel, patchFeatures)[0].label;
                    mostFreq.push_back(foodLabel);
                    //std::cout << foodLabel << " ";
                }
            }

            //std::cout << std::endl;
            std::vector<int> three = findThreeMostFrequent(mostFreq);
            //std::cout << three[0] << " " << three[1] << " " << three[2] << std::endl;

            int rows = std::ceil(croppedMask.rows / windowSize), cols = std::ceil(croppedMask.cols / windowSize);
            std::vector<std::vector<int>> labelMat(rows, std::vector<int> (cols)); 

            //std::cout << "three plates" << std::endl;

            for (int y = 0; y < croppedMask.rows - windowSize; y += windowSize) {
                int row = std::floor(float(y) / float(windowSize));
                for (int x = 0; x < croppedMask.cols - windowSize; x += windowSize) {
                    int col = std::floor(float(x) / float(windowSize));
                    cv::Rect windowRect(x, y, windowSize, windowSize);
                    cv::Mat submask = croppedMask(windowRect).clone();

                    cv::Mat otherMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(otherMask(bbox)(windowRect));

                    /*cv::Mat masked;
                    cv::bitwise_and(src, src, masked, otherMask);
                    cv::imshow("masked", masked);cv::waitKey();*/

                    cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, otherMask);
                    foodLabel = FeatureComparator::getLabelDistances(labels, three, patchFeatures)[0].label;
                    labelMat[row][col] = foodLabel;
                    //std::cout << labelMat[row][col] << " ";
                }
            }

            std::vector<std::vector<int>> second = createNewArray(labelMat);

            //std::cout << "median plates" << std::endl;

            for (int y = 0; y < croppedMask.rows - windowSize; y += windowSize) {
                int row = std::floor(float(y) / float(windowSize));
                for (int x = 0; x < croppedMask.cols - windowSize; x += windowSize) {
                    int col = std::floor(float(x) / float(windowSize));
                    cv::Rect windowRect(x, y, windowSize, windowSize);
                    cv::Mat submask = croppedMask(windowRect).clone();

                    cv::Mat otherMask = cv::Mat::zeros(tmpMask.size(), tmpMask.type());
                    submask.copyTo(otherMask(bbox)(windowRect));

                    /*cv::Mat masked;
                    cv::bitwise_and(src, src, masked, otherMask);
                    cv::imshow("masked", masked);cv::waitKey();*/

                    foodLabel = second[row][col];
                    //std::cout << second[row][col] << " ";

                    //updates the boundingbox file
                    //InsertBoundingBox(otherMask, foodLabel, filePath, file);

                    //create the segmented image
                    for(int r = 0; r < segmentationMask.rows; r++)
                        for(int c = 0; c < segmentationMask.cols; c++)
                            if(otherMask.at<uchar>(r,c) != 0)
                                segmentationMask.at<uchar>(r,c) = int(otherMask.at<uchar>(r,c) * foodLabel);
                }
            }
        }
    }

    file.close(); // Close the file

    return segmentationMask;
}

std::string ExtractName(std::string imagePath) {
    
    std::string imageName;
    bool lastOne = true;
    for(int i = imagePath.size(); i > 0; i--) {
        if(imagePath[i] == '/') {
            if (lastOne == false) {
                imageName = imagePath.substr(i+1, imagePath.size()-1);
                break;
            }
            lastOne = false;
        }
    }

    return "../output/" + imageName.substr(0, imageName.size()-4) + ".txt";
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
    std::ifstream file(filePath); // Replace with the actual file path
    
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

            cv::rectangle(out, topLeft, bottomRight, colors[foodId], 5);  // Draw the rectangle on the image
        }
        
        return out;
    }
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

    cv::waitKey();
}