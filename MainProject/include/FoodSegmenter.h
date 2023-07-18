#pragma once
#include <opencv2/opencv.hpp>

class FoodSegmenter {
    private:
        static const int SMALL_WINDOW_SIZE = 50;
        static const int BIG_WINDOW_SIZE = 150;
        static constexpr float PI = 3.141;
        static const std::vector<int> FIRST_PLATES_LABELS;
        static const std::vector<int> SECOND_PLATES_LABELS;
        static const std::vector<int> SIDE_DISHES_LABELS;
        
    public:
        static const int NUM_LABELS = 14;
        static const std::string LABEL_NAMES[14];
        static const int SALAD_LABEL = 12;
        static const int BREAD_LABEL = 13;

        static void getFoodMaskFromPlate(const cv::Mat&, cv::Mat&, cv::Vec3f);
        static void getFoodMaskFromPlates(const cv::Mat&, cv::Mat&, std::vector<cv::Vec3f> , std::vector<int>&);
        static void getSaladMaskFromBowl(const cv::Mat&, cv::Mat&, cv::Vec3f);
        static void getBreadMask(const cv::Mat&, const cv::Mat&, cv::Mat&);
        static void refinePestoPasta(const cv::Mat&, cv::Mat&);
        static void refineTomatoPasta(const cv::Mat&, cv::Mat&);
        static void refinePorkCutlet(const cv::Mat&, cv::Mat&);
        static void refineMask(const cv::Mat&, cv::Mat& , int);
        static void refinePillowRice(const cv::Mat&, cv::Mat&);
};