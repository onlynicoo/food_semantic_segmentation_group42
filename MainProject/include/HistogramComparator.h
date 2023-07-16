#pragma once
#include <opencv2/opencv.hpp>

class HistogramComparator {
    private:
        static void getHueHistograms(const cv::Mat&, const cv::Mat&, int, cv::Mat&);
        static void getLBPHistograms(const cv::Mat&, const cv::Mat&, int, cv::Mat&);
        static void getCannyLBPHistograms(const cv::Mat&, const cv::Mat&, int, cv::Mat&);

    public:
        struct LabelDistance {
            int label;
            double distance;

            bool operator<(const LabelDistance& other) const {
                return distance < other.distance;
            }
        };

        static const int NORMALIZE_VALUE = 100;
        static const std::string LABEL_HISTOGRAMS_PATH;
        static const std::string LABEL_HISTOGRAMS_NAME;

        static void getImageHistograms(const cv::Mat&, const cv::Mat&, cv::Mat&);
        static std::vector<LabelDistance> getLabelDistances(const cv::Mat&, std::vector<int>, const cv::Mat&);
        static void writeLabelHistogramsToFile(const cv::Mat&);
        static void readLabelHistogramsFromFile(cv::Mat&);
};