//#ifndef MY_SEGMENTATION_H // Header guard to prevent multiple inclusions
//#define MY_SEGMENTATION_H

#include <opencv2/opencv.hpp>

cv::Mat K_Means(const cv::Mat &, int);

std::vector<cv::Mat> extractRegionOfKMeans(const cv::Mat&);

std::vector<double> calculateCentroid(const std::vector<std::vector<double>>&);

double featureColorGaborL2Norm(const std::vector<double>&, const std::vector<double>&);

std::vector<cv::Mat> meanShiftGaborColorSegmentation(const std::vector<cv::Mat>&, const std::vector<std::vector<double>>&);

void mySeg(const cv::Mat& img);

//#endif