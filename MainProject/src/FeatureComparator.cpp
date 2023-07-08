#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/FeatureComparator.h"

std::vector<FeatureComparator::LabelDistance> FeatureComparator::getLabelDistances(
    cv::Mat labelsFeatures, std::vector<int> labelWhitelist, cv::Mat imgFeatures)
{
    std::vector<FeatureComparator::LabelDistance> distances;
    for (int i = 0; i < labelsFeatures.rows; i++) {

        // Check if the label is excluded
        if (std::find(labelWhitelist.begin(), labelWhitelist.end(), i) == labelWhitelist.end())
            continue;

        double distance = cv::norm(labelsFeatures.row(i), imgFeatures, cv::NORM_L2);
        LabelDistance labelDistance;
        labelDistance.label = i;
        labelDistance.distance = distance;
        distances.push_back(labelDistance);
    }
    std::sort(distances.begin(), distances.end());
    return distances;
}

cv::Mat FeatureComparator::getHueFeatures(cv::Mat img, cv::Mat mask, int numFeatures) {
    // Convert to HSV
    cv::Mat hsvImg;
    std::vector<cv::Mat> hsvChannels;
    cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV); 
    split(hsvImg, hsvChannels);

    // Equalize V channel to enhance color difference
    cv::Mat valueChannel;
    hsvChannels[2].copyTo(valueChannel, mask);
    cv::equalizeHist(valueChannel, valueChannel);
    valueChannel.copyTo(hsvChannels[2], mask);

    // Merge back the channels and convert back to BGR
    cv::Mat modHsvImg, modBgrImg;
    cv::merge(hsvChannels, modHsvImg);

    // Convert to HSV
    cv::Mat hueChannel;
    cvtColor(modHsvImg, hsvImg, cv::COLOR_BGR2HSV);
    cv::split(hsvImg, hsvChannels);
    hueChannel = hsvChannels[0];
    
    // Compute hist
    float range[] = {0, 180};
    const float* histRange[] = {range};
    cv::Mat hist;
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numFeatures, histRange);
    
    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    return hist.t();
}

cv::Mat FeatureComparator::getLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures) {
    // Compute LBP texture features
    int lbp_radius = 1;
    int lbp_neighbors = pow(lbp_radius * 2 + 1, 2) - 1;
    cv::Mat lbp_image = cv::Mat::zeros(img.size(), CV_8UC1);

    for (int y = lbp_radius; y < img.rows - lbp_radius; y++) {
        for (int x = lbp_radius; x < img.cols - lbp_radius; x++) {
            
            // Skip not maked pixels
            if (mask.at<uchar>(y, x) == 0)
                continue;

            uchar center_value = img.at<uchar>(y, x);
            uchar lbp_code = 0;
            for (int i = 0; i < lbp_neighbors; i++) {
                float angle = 2 * CV_PI * i / lbp_neighbors;
                int x_i = int(x + lbp_radius * cos(angle));
                int y_i = int(y + lbp_radius * sin(angle));

                uchar neighbor_value = img.at<uchar>(y_i, x_i);
                if (neighbor_value >= center_value)
                    lbp_code++;
            }

            lbp_image.at<uchar>(y, x) = lbp_code;
        }
    }

    // Compute hist
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::Mat hist;
    calcHist(&lbp_image, 1, 0, mask, hist, 1, &numFeatures, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);
    
    return hist.t();
}

cv::Mat FeatureComparator::getCannyLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures) {

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    // Blur image
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);

    // Apply Canny edge detection
    cv::Mat edges;
    int t1 = 50, t2 = 150;
    Canny(grayImage, edges, t1, t2);

    return getLBPFeatures(edges, mask, numFeatures);
}

void FeatureComparator::appendColumns(cv::Mat src, cv::Mat &dst) {
    if (dst.empty())
        dst = src;
    else
        cv::hconcat(src, dst, dst);
}

cv::Mat FeatureComparator::getImageFeatures(cv::Mat img, cv::Mat mask) {
    cv::Mat features;
    appendColumns(0.6 * getHueFeatures(img, mask, 64), features);
    appendColumns(0.4 * getCannyLBPFeatures(img, mask, 64), features);
    return features;
}