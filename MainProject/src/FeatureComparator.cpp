#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/FeatureComparator.h"

using namespace cv;
using namespace std;

int FeatureComparator::getFoodLabel(Mat labelsFeatures, vector<int> excludedLabels, Mat imgFeatures) {
    double minDistance = DBL_MAX;
    int nearestLabelIdx = -1;
    for (int i = 0; i < labelsFeatures.rows; i++) {
        
        // Check if the label is excluded
        if (find(excludedLabels.begin(), excludedLabels.end(), i) != excludedLabels.end())
            continue;

        Mat curFeatures = labelsFeatures.row(i);
        double distance = norm(curFeatures, imgFeatures, NORM_L2);
        if (distance < minDistance) {
            minDistance = distance;
            nearestLabelIdx = i;
        }
    }
    return nearestLabelIdx;
}

Mat FeatureComparator::getHueFeatures(Mat img, Mat mask, int numFeatures) {
    // Convert to HSV
    Mat hsvImg;
    vector<Mat> hsvChannels;
    cvtColor(img, hsvImg, COLOR_BGR2HSV); 
    split(hsvImg, hsvChannels);

    // Equalize V channel to enhance color difference
    Mat valueChannel;
    hsvChannels[2].copyTo(valueChannel, mask);
    equalizeHist(valueChannel, valueChannel);
    valueChannel.copyTo(hsvChannels[2], mask);

    // Merge back the channels and convert back to BGR
    Mat modHsvImg, modBgrImg;
    merge(hsvChannels, modHsvImg);

    // Convert to HSV
    Mat hueChannel;
    cvtColor(modHsvImg, hsvImg, COLOR_BGR2HSV);
    split(hsvImg, hsvChannels);
    hueChannel = hsvChannels[0];
    
    // Compute hist
    float range[] = {0, 180};
    const float* histRange[] = {range};
    Mat hist;
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numFeatures, histRange);
    
    // Normalize the hist
    hist /= sum(hist)[0];

    return hist.t();
}

Mat FeatureComparator::getLBPFeatures(Mat img, Mat mask, int numFeatures) {
    // Compute LBP texture features
    int lbp_radius = 1;
    int lbp_neighbors = pow(lbp_radius * 2 + 1, 2) - 1;
    Mat lbp_image = Mat::zeros(img.size(), CV_8UC1);

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
    Mat hist;
    calcHist(&lbp_image, 1, 0, mask, hist, 1, &numFeatures, histRange);

    // Normalize the hist
    hist /= sum(hist)[0];
    
    return hist.t();
}

Mat FeatureComparator::getCannyLBPFeatures(Mat img, Mat mask, int numFeatures) {

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);

    // Blur image
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);

    // Apply Canny edge detection
    Mat edges;
    int t1 = 50, t2 = 150;
    Canny(grayImage, edges, t1, t2);

    return getLBPFeatures(edges, mask, numFeatures);
}

void FeatureComparator::appendColumns(Mat src, Mat &dst) {
    if (dst.empty())
        dst = src;
    else
        hconcat(src, dst, dst);
}

Mat FeatureComparator::getImageFeatures(Mat img, Mat mask) {
    Mat features;
    appendColumns(0.6 * getHueFeatures(img, mask, 64), features);
    appendColumns(0.4 * getCannyLBPFeatures(img, mask, 64), features);
    return features;
}