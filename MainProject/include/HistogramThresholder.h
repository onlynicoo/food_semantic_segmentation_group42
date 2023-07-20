#pragma once
#include <opencv2/opencv.hpp>

class HistogramThresholder {
    private:
        
        // histogram normalization value
        static const int NORMALIZE_VALUE = 100;

        // it computes hue histogram
        static void getHueHistogram(const cv::Mat&, const cv::Mat&, cv::Mat&);

    public:
        struct LabelDistance {
            int label;
            double distance;

            bool operator<(const LabelDistance& other) const {
                return distance < other.distance;
            }
        };

        static const int NUM_VALUES = 60;
        static const int DATA_TYPE = CV_8UC1;
        static const std::string LABELS_HISTOGRAMS_PATH;
        static const std::string LABELS_HISTOGRAMS_NAME;

        // calls getHueHistogram for a given image patch
        static void getImageHistogram(const cv::Mat&, const cv::Mat&, cv::Mat&);

        // computes the distances between given image histogram and the labels ones
        static std::vector<LabelDistance> getLabelDistances(const cv::Mat&, std::vector<int>, const cv::Mat&);

        // 
        static void readLabelsHistogramsFromFile(cv::Mat&);
};