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
                if (hsv[1] > 80 || (hsv[0] > 20 && hsv[0] < 25))
                    mask.at<int8_t>(cur) = 1;
            }
        }

    // Open weak connected components
    /*int openingSize = radius / 30;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openingSize, openingSize));
    morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);*/
    
    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Check if there is a big area and remove the small areas (w.r.t. the plate area)
    double plateArea = PlateRemover::PI * std::pow(radius, 2);
    bool foundBigArea = false;
    for(int i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area > plateArea * 0.07)
            foundBigArea = true;
        else if (area < plateArea * 0.005) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    std::vector<std::vector<cv::Point>> keptContours;

    // Sort contours based on area
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::contourArea(contour1) > cv::contourArea(contour2);
    });

    if (foundBigArea) {

        // If there is a big area, keep only that one
        int n = 1;
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));

    } else {
        
        // Otherwise, keep the two biggest if they are not too far from the center (w.r.t. the plate radius)
        int n = 2;
        keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));
        for(int i = 0; i < keptContours.size(); i++) {
            double distance = std::abs(cv::pointPolygonTest(keptContours[i], center, true));
            // std::cout << "Distance " << distance << std::endl;
            if (distance > radius * 0.75) {
                keptContours.erase(keptContours.begin() + i);
                i--;
            }
        }
    }

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, keptContours, -1, cv::Scalar(1), cv::FILLED);

    // Fill the holes
    int closingSize = radius / 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

void PlateRemover::getSaladMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius) {

    // Pre-process
    cv::Mat img;
    cv::GaussianBlur(src, img, cv::Size(5,5), 0);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    // Find salad mask
    mask = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    for (int r = std::max(0, center.y - radius); r < std::min(center.y + radius + 1, img.rows); r++)
        for (int c = std::max(0, center.x - radius); c < std::min(center.x + radius + 1, img.cols); c++) {
            cv::Point cur = cv::Point(c, r);
            if (cv::norm(cur - center) <= radius) {

                // Check if current point is not part of the bowl
                int hsv[3] = {int(img.at<cv::Vec3b>(cur)[0]), int(img.at<cv::Vec3b>(cur)[1]), int(img.at<cv::Vec3b>(cur)[2])};
                if (hsv[1] > 175 || hsv[2] > 245)
                    mask.at<int8_t>(cur) = 1;
            }
        }

    // Fill the holes
    int closingSize = radius / 3;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Remove the small areas (w.r.t. the bowl area)
    double bowlArea = PlateRemover::PI * std::pow(radius, 2);
    for(int i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area < bowlArea * 0.001) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, contours, -1, cv::Scalar(1), cv::FILLED);
}