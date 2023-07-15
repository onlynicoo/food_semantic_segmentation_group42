#pragma once
#include <opencv2/opencv.hpp>

class SegmentFood {
    private:
        static constexpr float PI = 3.141;
        
    public:
        static void getFoodMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius);
        static void getSaladMask(cv::Mat src, cv::Mat &mask, cv::Point center, int radius);
        static cv::Mat SegmentBread(cv::Mat src);
        static void refinePestoPasta(const cv::Mat& src, cv::Mat& mask);
        static cv::Mat refineTomatoPasta(cv::Mat src, cv::Mat mask);
        static void refinePorkCutlet(cv::Mat src, cv::Mat &mask);
        static void refineMask(const cv::Mat& src, cv::Mat& mask, int label);
};