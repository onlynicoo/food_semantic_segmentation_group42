#include "../include/Tray.h"
/*
Tray::Tray() {
    traysNumber = 0;
}
*/
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

void Tray::ElaborateImage(const cv::Mat src, cv::Mat tmpDest[2]) {
    // it contains
    // image detection | image segmentation
    
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    

    tmpDest[0] = src;
    tmpDest[1] = src;
    // ... add code ...
}

Tray::Tray(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysNumber = 1;

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;
    
    traysBefore = before;
    traysAfter = after;

    cv::Mat tmpDest[2];
    ElaborateImage(before, tmpDest);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];
    
    ElaborateImage(after, tmpDest);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];
}
/*
void Tray::AddTray(std::string trayBefore, std::string trayAfter) {

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;
    
    traysBefore = before;
    traysAfter = after;

    cv::Mat tmpDest[2];
    ElaborateImage(before, tmpDest);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];
    
    ElaborateImage(after, tmpDest);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];

    traysNumber ++;    
}
*/
void Tray::PrintInfo() {
    std::cout << "The number of insterted trays is: " << traysNumber << std::endl;
}


