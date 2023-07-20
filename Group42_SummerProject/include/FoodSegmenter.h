// Author: Nicola Lorenzon, Daniele Moschetta

#pragma once
#include <opencv2/opencv.hpp>

// Handles the segmenting of plates, salad bowl and bread
class FoodSegmenter {
    private:

        // Constants
        static const int SMALL_WINDOW_SIZE = 50;
        static const int BIG_WINDOW_SIZE = 150;
        static constexpr float PI = 3.141;

        // First dishes labels
        static const std::vector<int> FIRST_PLATES_LABELS;

        // Second dishes labels
        static const std::vector<int> SECOND_PLATES_LABELS;

        // Side dishes labels
        static const std::vector<int> SIDE_DISHES_LABELS;
        
    public:

        // Number of labels
        static const int NUM_LABELS = 14;

        // Names of labels
        static const std::string LABEL_NAMES[14];

        // Salad label
        static const int SALAD_LABEL = 12;

        // Bread label
        static const int BREAD_LABEL = 13;

        // Finds food mask from a plate
        static void getFoodMaskFromPlate(const cv::Mat&, cv::Mat&, cv::Vec3f);

        // Segments the foods found in a plate
        static void getFoodMaskFromPlates(const cv::Mat&, cv::Mat&, std::vector<cv::Vec3f> , std::vector<int>&);

        // Finds the salad mask from a salad bowl
        static void getSaladMaskFromBowl(const cv::Mat&, cv::Mat&, cv::Vec3f);

        // Finds the bread mask from the bread area
        static void getBreadMask(const cv::Mat&, const cv::Mat&, cv::Mat&);

        // Refines pasta with pesto
        static void refinePestoPasta(const cv::Mat&, cv::Mat&);

        // Refines pasta with tomato
        static void refineTomatoPasta(const cv::Mat&, cv::Mat&);

        // Refines pork cutlet
        static void refinePorkCutlet(const cv::Mat&, cv::Mat&);

        // Refines pilaw rice
        static void refinePilawRice(const cv::Mat&, cv::Mat&);

        // Entrypoint function that calls the correct refine function based on the label
        static void refineMask(const cv::Mat&, cv::Mat& , int);
};