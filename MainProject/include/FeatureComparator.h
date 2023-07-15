#pragma once
#include <opencv2/opencv.hpp>

class FeatureComparator {
    private:
        static cv::Mat getHueFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static cv::Mat getLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static cv::Mat getCannyLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        static void appendColumns(cv::Mat src, cv::Mat &dst);

    public:
        struct LabelDistance {
            int label;
            double distance;

            bool operator<(const LabelDistance& other) const {
                return distance < other.distance;
            }
        };

        static const int NORMALIZE_VALUE = 100;

        static cv::Mat getImageFeatures(cv::Mat img, cv::Mat mask);
        static std::vector<LabelDistance> getLabelDistances(cv::Mat labelsFeatures, std::vector<int> labelWhitelist, cv::Mat imgFeatures);
};