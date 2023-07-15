#include <opencv2/opencv.hpp>
#include "../include/FindFood.h"
#include "../include/SegmentFood.h"

void SegmentFood::getFoodMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius) {

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
    double plateArea = SegmentFood::PI * std::pow(radius, 2);
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

void SegmentFood::getSaladMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius) {

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
    double bowlArea = SegmentFood::PI * std::pow(radius, 2);
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

cv::Mat SegmentFood::SegmentBread(cv::Mat src) {
    
    cv::Mat breadMask = FindFood::findBread(src);

    int kernelSizeErosion = 5; // Adjust the size according to your needs
    cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeErosion, kernelSizeErosion));
    cv::Mat erodedMask;
    cv::morphologyEx(breadMask, erodedMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 3);

    cv::Rect bbx = cv::boundingRect(erodedMask);
    if (bbx.empty())
        return cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat final_image, result_mask, bgModel, fgModel;

    // GrabCut segmentation algorithm for current box
    grabCut(src,		// input image
        result_mask,	// segmentation resulting mask
        bbx,			// rectangle containing foreground
        bgModel, fgModel,	// models
        10,				// number of iterations
        cv::GC_INIT_WITH_RECT);
    
    cv::Mat tmpMask0 = (result_mask == 0);
    cv::Mat tmpMask1 = (result_mask == 1);
    cv::Mat tmpMask2 = (result_mask == 2);
    cv::Mat tmpMask3 = (result_mask == 3);

    cv::Mat foreground = (tmpMask3 | tmpMask1);

    // Perform connected component analysis
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat out = cv::Mat::zeros(foreground.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestAreaPost) {
            largestAreaPost = area;
            index = i;
        }    
    }
    if (index != -1) 
        cv::fillPoly(out, contours[index], cv::Scalar(13));

    return out;
}   

void SegmentFood::refinePestoPasta(const cv::Mat& src, cv::Mat& mask) {
    cv::Mat workingFood;
    cv::Mat helperMask = mask.clone();
    cv::Mat fullSizeMask(src.rows, src.cols, CV_8UC1, cv::Scalar(1));

    cv::bitwise_and(src, src, workingFood, helperMask);

    //cv::imshow("tmp1", workingFood); cv::waitKey();

    cv::Mat hsvImage;
    cv::cvtColor(workingFood, hsvImage, cv::COLOR_BGR2HSV);
    
    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    cv::Mat thresholdingMask = hsvChannels[1] > 0.6*255;
    cv::Mat thresholdedImage;
    cv::bitwise_and(src, src, thresholdedImage, thresholdingMask);

    //cv::imshow("thresholdedImage", thresholdedImage);

    // Access and process the Y, U, and V channels separately
    std::vector<cv::Mat> thresholdedImageChannels;
    cv::split(thresholdedImage, thresholdedImageChannels);
    cv::Mat thresholdedGreenMask = thresholdingMask;

    for (int i = 0; i < thresholdingMask.rows; i++) {
        for (int j = 0; j < thresholdingMask.cols; j++) {
            if (thresholdedImageChannels[1].at<uchar>(i,j) < 85)
                thresholdedGreenMask.at<uchar>(i,j) = 0;
        }
    }

    int kernelSizeDilation = 5;
    cv::Mat kernelDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeDilation, kernelSizeDilation));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat dilatedMask;
    cv::morphologyEx(thresholdedGreenMask, dilatedMask, cv::MORPH_DILATE, kernelDilation, cv::Point(-1, -1), 1);

    int kernelSizeClosure = 3;
    cv::Mat kernelClosure = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosure, kernelSizeClosure));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat closedMask;
    cv::morphologyEx(dilatedMask, closedMask, cv::MORPH_CLOSE, kernelClosure, cv::Point(1, 1), 10);

    int kernelSizeOpening = 3;
    cv::Mat kernelOpening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeOpening, kernelSizeOpening));

    // Perform dilation followed by erosion (closing operation)
    cv::Mat openMask;
    cv::morphologyEx(closedMask, openMask, cv::MORPH_OPEN, kernelOpening, cv::Point(1, 1), 2);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(openMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(openMask.size(), CV_8UC1);


    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }

    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(1));
    
    mask = filledMask;
}

cv::Mat SegmentFood::refineTomatoPasta(cv::Mat src, cv::Mat mask) {
    cv::Mat workingFood;
    cv::bitwise_and(src, src, workingFood, mask);
    // cv::imshow("workingFood", workingFood);  cv::waitKey();
    
    // Split the image into individual BGR channels
    std::vector<cv::Mat> channels;
    cv::split(workingFood, channels);

    // Keep only the red channel
    cv::Mat redChannel = channels[2];

    cv::Mat thresholdedRedChannel = redChannel > 160;
    cv::Mat bgrThresholded;
    cv::bitwise_and(workingFood, workingFood, bgrThresholded, thresholdedRedChannel);
    
    //cv::imshow("thresholdedRedChannel", thresholdedRedChannel);


    int kernelSizeClosing = 3;
    cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeClosing, kernelSizeClosing));
    // Perform dilation followed by erosion (closing operation)
    cv::Mat closingMask;
    cv::morphologyEx(thresholdedRedChannel, closingMask, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 5);
    cv::Mat closingImage;
    cv::bitwise_and(workingFood, workingFood, closingImage, closingMask);

    //cv::imshow("closingMask", closingMask);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closingMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(closingMask.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }

    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(255));

    //cv::imshow("filledMask", filledMask);
    //cv::imshow("bgrThresholded", bgrThresholded);

    //cv::waitKey();
    return bgrThresholded ;
}

void SegmentFood::refinePorkCutlet(cv::Mat src, cv::Mat &mask) {
        
    // Close the holes borders
    int closingSize = cv::boundingRect(mask).height / 4;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closingSize, closingSize));
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Sort contours based on area
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::contourArea(contour1) > cv::contourArea(contour2);
    });

    int n = 1;
    std::vector<std::vector<cv::Point>> keptContours;
    keptContours = std::vector<std::vector<cv::Point>>(contours.begin(), contours.begin() + std::min(n, (int) contours.size()));

    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(mask, keptContours, -1, cv::Scalar(1), cv::FILLED);
}

void SegmentFood::refineMask(const cv::Mat& src, cv::Mat& mask, int label) {
    switch (label) {
        case 1:
            refinePestoPasta(src, mask);
            break;
        case 2:
            refineTomatoPasta(src, mask);
            break;
        case 6:
            refinePorkCutlet(src, mask);
            break;
        default:
            break;
    }
}