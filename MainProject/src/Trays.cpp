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

void Tray::ElaborateImage(const cv::Mat src, cv::Mat tmpDest[2]) {
    // it contains
    // image detection | image segmentation
    
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    

    tmpDest[0] = PlatesFinder::print_plates_image(src, plates);

    tmpDest[1] = src;
    // ... add code ...
}

Tray::Tray(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

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
    tmp2_1.copyTo(imageRow);
    hconcat(tmp2_2, imageRow, imageRow);
    hconcat(tmp2_3, imageRow, imageRow);
    imageRow.copyTo(imageGrid);
    imageRow.release();

    tmp1_1.copyTo(imageRow);
    hconcat(tmp1_2, imageRow, imageRow);
    hconcat(tmp1_3, imageRow, imageRow);
    vconcat(imageRow, imageGrid, imageGrid);
    imageRow.release();

    // Resize the full image grid and display it
    resize(imageGrid, imageGrid, stdSize);
    imshow(window_name_before, imageGrid);

    cv::waitKey();
    std::cout << "The number of insterted trays is: " << traysNumber << std::endl;


}


