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

Trays::Trays(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysNumber = 1;

    traysBeforeNames.push_back(trayBefore);
    traysAfterNames.push_back(trayAfter);
    
    traysBefore.push_back(before);
    traysAfter.push_back(after);

    traysBeforeDetected.push_back(DetectFoods(before));
    traysAfterDetected.push_back(DetectFoods(after));

    traysBeforeSegmented.push_back(SegmentFoods(before));
    traysBeforeSegmented.push_back(SegmentFoods(after));
    
}

void Trays::AddTray(std::string trayBefore, std::string trayAfter) {

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayBefore, cv::IMREAD_COLOR);

    traysBeforeNames.push_back(trayBefore);
    traysAfterNames.push_back(trayAfter);
    
    traysBefore.push_back(before);
    traysAfter.push_back(after);

    traysBeforeDetected.push_back(DetectFoods(before));
    traysAfterDetected.push_back(DetectFoods(after));

    traysBeforeSegmented.push_back(SegmentFoods(before));
    traysBeforeSegmented.push_back(SegmentFoods(after));
    
    traysNumber ++;    
}

void Trays::PrintInfo() {
    std::cout << "The number of insterted trays is: " << traysNumber << std::endl;
}


