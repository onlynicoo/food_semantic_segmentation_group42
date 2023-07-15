#pragma once
#include <opencv2/opencv.hpp>

class SegmentFood {
    private:
        static constexpr int SMALL_WINDOW_SIZE = 50;
        static constexpr int BIG_WINDOW_SIZE = 150;
        static constexpr float PI = 3.141;
        
    public:
        static const int SALAD_LABEL = 12;
        static const int BREAD_LABEL = 13;
        static const std::vector<int> FIRST_PLATES_LABELS;
        static const std::vector<int> SECOND_PLATES_LABELS;
        static const std::vector<int> SIDE_DISHES_LABELS;
        static const std::string LABEL_NAMES[14];

        static void getFoodMaskFromPlate(cv::Mat src, cv::Mat &mask, cv::Vec3f plate);
        static void getFoodMaskFromPlates(cv::Mat src, cv::Mat &mask, std::vector<cv::Vec3f> plates, std::vector<int> labelsFound);
        static void getSaladMaskFromBowl(cv::Mat src, cv::Mat &mask, cv::Vec3f bowl);
        static cv::Mat getBreadMask(cv::Mat src, cv::Mat breadMask);
        static void refinePestoPasta(const cv::Mat& src, cv::Mat& mask);
        static cv::Mat refineTomatoPasta(cv::Mat src, cv::Mat mask);
        static void refinePorkCutlet(cv::Mat src, cv::Mat &mask);
        static void refineMask(const cv::Mat& src, cv::Mat& mask, int label);
};