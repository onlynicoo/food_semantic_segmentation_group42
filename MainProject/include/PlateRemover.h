#pragma once
#include <opencv2/opencv.hpp>

class PlateRemover {
    private:
        static constexpr float PI = 3.141;

    public:
        static void getFoodMask(cv::Mat img, cv::Mat &mask, cv::Point center, int radius);
        static void getSaladMask(cv::Mat img, cv::Mat &mask, cv::Point center, int radius);
};