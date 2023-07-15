#include <opencv2/opencv.hpp>
#include "../include/FindFood.h"

// Find plate image
std::vector<cv::Vec3f> FindFood::findPlates(cv::Mat src) {

    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
 
    // Find the circle
    std::vector<cv::Vec3f> circles_plate;
    std::vector<cv::Vec3f> circles_salad;
    std::vector<cv::Vec3f> actual_plates;
    std::vector<cv::Vec3f> refine_salad;

    HoughCircles(src_gray, circles_plate, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_plates, max_radius_hough_plates);
    HoughCircles(src_gray, circles_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_salad, max_radius_hough_salad);
    HoughCircles(src_gray, refine_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_refine, max_radius_refine);

    actual_plates = circles_plate;

    // Remove salad circles
    for (size_t i = 0; i < circles_salad.size(); i++) {
        for (size_t j = 0; j < circles_plate.size(); j++) {
            cv::Vec3i s = circles_salad[i];
            cv::Vec3i p = circles_plate[j];
            cv::Point center_salad = cv::Point(s[0], s[1]);
            cv::Point center_plate = cv::Point(p[0], p[1]);
            if (cv::norm(center_plate - center_salad) < p[2]) {
                std::vector<cv::Vec3f>::iterator it = actual_plates.begin()+j;
                actual_plates.erase(it);
            }
        }
    }
    
    if (actual_plates.size() > 2) {
        // Remove salad circles
        for (size_t i = 0; i < actual_plates.size(); i++) {
            for (size_t j = 0; j < refine_salad.size(); j++) {
                cv::Vec3i s = actual_plates[i];
                cv::Vec3i p = refine_salad[j];
                cv::Point center_salad = cv::Point(s[0], s[1]);
                cv::Point center_plate = cv::Point(p[0], p[1]);
                if (cv::norm(center_plate - center_salad) < p[2]) {
                    std::vector<cv::Vec3f>::iterator it = actual_plates.begin() + i;
                    actual_plates.erase(it);
                }
            }
        }
    }

    return actual_plates;

}

// Print plates in image
cv::Mat FindFood::drawPlates(const cv::Mat src, const std::vector<cv::Vec3f> circles) {
    cv::Mat output = src.clone();
    // Draw the circle
    for( size_t i = 0; i < circles.size(); i++ ) {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle outline
        int radius = c[2];
        circle(output, center, radius, cv::Scalar(255,0,0), 20);
    }
    return output;
}

std::vector<cv::Vec3f> FindFood::findSaladBowl(cv::Mat src, bool saladFound) {

    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
 
    // Find the circle
    std::vector<cv::Vec3f> circles_salad;

    HoughCircles(src_gray, circles_salad, cv::HOUGH_GRADIENT, 1, src.rows/ratioMinDist, param1, param2, min_radius_hough_salad, max_radius_hough_salad);
    
    if (circles_salad.size() == 1 || !saladFound) {
            return circles_salad;
    }

    else {
        std::vector<cv::Vec3f> circles_salad_refined;
        HoughCircles(src_gray, circles_salad_refined, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, paramSalad1, paramSalad2, min_radius_hough_salad_refine, max_radius_hough_salad_refine);
        
        std::vector<cv::Vec3f> toRemove = findPlates(src);

        std::vector<cv::Vec3f> actual_plates = circles_salad_refined;

        if (actual_plates.size() > 1) {
            // Remove salad circles
            for (size_t i = 0; i < actual_plates.size(); i++) {
                for (size_t j = 0; j < toRemove.size(); j++) {
                    cv::Vec3i s = actual_plates[i];
                    cv::Vec3i p = toRemove[j];
                    cv::Point center_salad = cv::Point(s[0], s[1]);
                    cv::Point center_toRemove = cv::Point(p[0], p[1]);
                    if (cv::norm(center_toRemove - center_salad) < p[2]) {
                        std::vector<cv::Vec3f>::iterator it = actual_plates.begin() + i;
                        actual_plates.erase(it);
                    }
                }
            }
        }
        return actual_plates;
    }
}

cv::Mat FindFood::findBread(cv::Mat src) {

    // used as base img
    cv::Mat maskedImage = src.clone();

    std::vector<cv::Vec3f> plates = FindFood::findPlates(src);
    std::vector<cv::Vec3f> salad = FindFood::findSaladBowl(src, true);

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
    cv::Mat out = cv::Mat::zeros(newMask.size(), CV_8UC1);

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
        cv::fillPoly(out, contours[index], cv::Scalar(13));

    return out;
}