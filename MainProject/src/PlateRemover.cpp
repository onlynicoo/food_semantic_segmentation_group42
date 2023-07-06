#include <opencv2/opencv.hpp>
#include "../include/PlateRemover.h"

void PlateRemover::getFoodMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius) {

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5,5), 0);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    // Find food mask
    mask = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    for (int r = std::max(0, center.y - radius); r < std::min(center.y + radius + 1, img.rows); r++)
        for (int c = std::max(0, center.x - radius); c < std::min(center.x + radius + 1, img.cols); c++) {
            cv::Point cur = cv::Point(c, r);
            if (cv::norm(cur - center) <= radius) {
                // Check if current point is not part of the plate
                int hsv[3] = {int(img.at<cv::Vec3b>(cur)[0]), int(img.at<cv::Vec3b>(cur)[1]), int(img.at<cv::Vec3b>(cur)[2])};
                if (hsv[1] > 80)
                    // changed from 255 to 1 for fast multiplications
                    mask.at<int8_t>(cur) = 1;
            }
        }

    // Fill the holes
    int closingSize = radius / 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Find the contour with the largest area
    double largestArea = DBL_MIN;
    int largestContourIndex = -1;
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestContourIndex = i;
        }
    }
    
    // Create a new mask with only the largest contour
    if (largestContourIndex != -1) {
        mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(mask, contours, largestContourIndex, cv::Scalar(255), cv::FILLED);
    }
}