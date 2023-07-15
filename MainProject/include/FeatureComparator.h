#pragma once
#include <opencv2/opencv.hpp>

class FeatureComparator {
    private:
        static void getHueFeatures(const cv::Mat&, const cv::Mat&, int, cv::Mat&);
        static void getLBPFeatures(const cv::Mat&, const cv::Mat&, int, cv::Mat&);
        static void getCannyLBPFeatures(const cv::Mat&, const cv::Mat&, int, cv::Mat&);

    public:
        struct LabelDistance {
            int label;
            double distance;

            bool operator<(const LabelDistance& other) const {
                return distance < other.distance;
            }
        };

        static const int NORMALIZE_VALUE = 100;
        static const std::string LABEL_FEATURES_PATH;
        static const std::string LABEL_FEATURES_NAME;

        static void getImageFeatures(const cv::Mat&, const cv::Mat&, cv::Mat&);
        static std::vector<LabelDistance> getLabelDistances(const cv::Mat&, std::vector<int>, const cv::Mat&);
        static void writeLabelFeaturesToFile(const cv::Mat&);
        static void readLabelFeaturesFromFile(cv::Mat&);
};