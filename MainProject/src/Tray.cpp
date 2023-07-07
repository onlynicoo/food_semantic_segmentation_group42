#include "../include/Tray.h"

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

void Tray::ElaborateImage(const cv::Mat src, cv::Mat tmpDest[2], std::vector<int>& labelsFound) {
    // it contains
    // image detection | image segmentation
    std::string LABELS[14] = {
        "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
        "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
        "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

    std::string labelFeaturesPath = "../data/label_features.yml";
    std::vector<int> excludedLabels = {0, 12, 13};
    if(labelsFound.size() != 0) {
        for(int i = 0; i < 14; i ++) {
            if(std::find(std::begin(labelsFound), std::end(labelsFound), i) == std::end(labelsFound)) 
                if(std::find(std::begin(excludedLabels), std::end(excludedLabels), i) == std::end(excludedLabels))
                    excludedLabels.push_back(i);
        }
    }

    int firstPlatesLabel[] = {1, 2, 3, 4, 5};
    
    std::map<int, cv::Vec3b> colors = InitColorMap();

    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentationMask(segmentationMaskSize, CV_8UC3, cv::Scalar(0));

    cv::Mat labels = GetTrainedFeatures(labelFeaturesPath);
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    plates.resize(std::min(2, (int) plates.size()));                          // Assume there are at most 2 plates
    tmpDest[0] = PlatesFinder::print_plates_image(src, plates);

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

        platesLabelDistances.push_back(FeatureComparator::getLabelDistances(labels, excludedLabels, patchFeatures));
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
            labelsFound.push_back(foodLabel);
            for(int r = 0; r < segmentationMask.rows; r++)
                for(int c = 0; c < segmentationMask.cols; c++)
                    if(platesMasks[i].at<uchar>(r,c) != 0)
                        segmentationMask.at<cv::Vec3b>(r,c) = colors[int(platesMasks[i].at<uchar>(r,c)*foodLabel)];
        }
        else {
            //add refinition of segmentation mask for multifood plates
            labelsFound.push_back(foodLabel);
        }
    }
    
    tmpDest[1] = segmentationMask;
    // ... add code ...
}

Tray::Tray(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;
    
    traysBefore = before;
    traysAfter = after;

    cv::Mat tmpDest[2];
    std::vector<int> labelsFound;
    ElaborateImage(before, tmpDest, labelsFound);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];
    
    ElaborateImage(after, tmpDest, labelsFound);
    traysAfterDetected = tmpDest[0];
    traysAfterSegmented = tmpDest[1];
}

void Tray::PrintInfo() {
    std::string window_name_before = "info tray";

    cv::Mat imageGrid, imageRow;
    cv::Size stdSize(0,0);
    stdSize = traysBefore.size();

    cv::Mat tmp1_1, tmp1_2, tmp1_3, tmp2_1, tmp2_2, tmp2_3; 
    tmp1_1 = traysBefore.clone();

    // Resize output to have all images of same size
    resize(traysAfter, tmp2_1, stdSize);
    resize(traysBeforeDetected, tmp1_2, stdSize);
    resize(traysAfterDetected, tmp2_2, stdSize);
    resize(traysBeforeSegmented, tmp1_3, stdSize);
    resize(traysAfterSegmented, tmp2_3, stdSize);

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
    imshow(window_name_before, imageGrid);

    cv::waitKey();
}