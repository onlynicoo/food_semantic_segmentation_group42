#include <opencv2/opencv.hpp>

int findNearestCenter(cv::Mat centers, cv::Mat dataRow);
cv::Mat getHistImage(cv::Mat hist);
cv::Mat getHueFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
cv::Mat getLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
cv::Mat getCannyLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
cv::Mat getSIFTFeatures(cv::Mat img, cv::Mat mask);