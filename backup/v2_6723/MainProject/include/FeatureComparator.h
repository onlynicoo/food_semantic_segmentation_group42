#include <iostream>
#include <opencv2/opencv.hpp>

class FeatureComparator {
    private:       
        static cv::Mat getHueFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static cv::Mat getLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static cv::Mat getCannyLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static void appendColumns(cv::Mat src, cv::Mat &dst);

    public:
        static cv::Mat getImageFeatures(cv::Mat img, cv::Mat mask);
        static int getFoodLabel(cv::Mat labelsFeatures, std::vector<int> excludedLabels, cv::Mat imgFeatures);
};