#pragma once
#include <opencv2/opencv.hpp>

class FoodSegmenter {
    private:

        // constants
        static const int SMALL_WINDOW_SIZE = 50;
        static const int BIG_WINDOW_SIZE = 150;
        static constexpr float PI = 3.141;

        // all the labels for first plates
        static const std::vector<int> FIRST_PLATES_LABELS;

        // all the labels for second plates
        static const std::vector<int> SECOND_PLATES_LABELS;

        // all the labels for side dishes
        static const std::vector<int> SIDE_DISHES_LABELS;
        
    public:

        // how many labels, the system must recognize
        static const int NUM_LABELS = 14;

        // all the notable labels
        static const std::string LABEL_NAMES[14];

        // salad label
        static const int SALAD_LABEL = 12;

        // bread label
        static const int BREAD_LABEL = 13;

        // function for retrieving food masks
        static void getFoodMaskFromPlate(const cv::Mat&, cv::Mat&, cv::Vec3f);
        static void getFoodMaskFromPlates(const cv::Mat&, cv::Mat&, std::vector<cv::Vec3f> , std::vector<int>&);

        // given a salad bowl, it returns the mask
        static void getSaladMaskFromBowl(const cv::Mat&, cv::Mat&, cv::Vec3f);

        // it returns the bread mask
        static void getBreadMask(const cv::Mat&, const cv::Mat&, cv::Mat&);

        // function for refining some food items
        static void refinePestoPasta(const cv::Mat&, cv::Mat&);
        static void refineTomatoPasta(const cv::Mat&, cv::Mat&);
        static void refinePorkCutlet(const cv::Mat&, cv::Mat&);
        static void refineMask(const cv::Mat&, cv::Mat& , int);
        static void refinePilawRice(const cv::Mat&, cv::Mat&);
};