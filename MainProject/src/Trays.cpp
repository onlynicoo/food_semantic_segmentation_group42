#include "../include/Trays.h"

Trays::Trays() {
    traysNumber = 0;
}

cv::Mat Trays::DetectFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
}

cv::Mat Trays::SegmentFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
}

void Trays::ElaborateImage(const cv::Mat src, cv::Mat tmpDest[2]) {
    // it contains
    // image detection | image segmentation
    
    tmpDest[0] = src;
    tmpDest[1] = src;
    // ... add code ...
}

Trays::Trays(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysNumber = 1;

    traysBeforeNames.push_back(trayBefore);
    traysAfterNames.push_back(trayAfter);
    
    traysBefore.push_back(before);
    traysAfter.push_back(after);

    cv::Mat tmpDest[2];
    ElaborateImage(before, tmpDest);
    traysBeforeDetected.push_back(tmpDest[0]);

    traysBeforeSegmented.push_back(tmpDest[1]);
    
    ElaborateImage(after, tmpDest);
    traysBeforeDetected.push_back(tmpDest[0]);
    traysBeforeSegmented.push_back(tmpDest[1]);
}

void Trays::AddTray(std::string trayBefore, std::string trayAfter) {

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysBeforeNames.push_back(trayBefore);
    traysAfterNames.push_back(trayAfter);
    
    traysBefore.push_back(before);
    traysAfter.push_back(after);

    cv::Mat tmpDest[2];
    ElaborateImage(before, tmpDest);
    traysBeforeDetected.push_back(tmpDest[0]);
    traysBeforeSegmented.push_back(tmpDest[1]);
    
    ElaborateImage(after, tmpDest);
    traysBeforeDetected.push_back(tmpDest[0]);
    traysBeforeSegmented.push_back(tmpDest[1]);

    traysNumber ++;    
}

void Trays::PrintInfo() {
    std::cout << "The number of insterted trays is: " << traysNumber << std::endl;
}


