#pragma once
#include <opencv2/opencv.hpp>

class HistogramThresholder {
    private:
        
        static const int NORMALIZE_VALUE = 100;

        static void getHueHistogram(const cv::Mat&, const cv::Mat&, cv::Mat&);

    public:
        struct LabelDistance {
            int label;
            double distance;

            bool operator<(const LabelDistance& other) const {
                return distance < other.distance;
            }
        };

        static const int NUM_VALUES = 32;
        static const int DATA_TYPE = CV_8UC1;
        static const std::string LABELS_HISTOGRAMS_PATH;
        static const std::string LABELS_HISTOGRAMS_NAME;

        static void getImageHistogram(const cv::Mat&, const cv::Mat&, cv::Mat&);
        static std::vector<LabelDistance> getLabelDistances(const cv::Mat&, std::vector<int>, const cv::Mat&);
        static void writeLabelsHistogramsToFile(const cv::Mat&);
        static void readLabelsHistogramsFromFile(cv::Mat&);
};