// Author: Nicola Lorenzon

#pragma once
#include <opencv2/opencv.hpp>

// Handles the finding of plates, salad bowl and bread
class FoodFinder {
    private:
        // Attributes

        // Radius parameters to detect plates
        static const int min_radius_hough_plates = 280;
        static const int max_radius_hough_plates = 300;
        
        // Radius parameters to detect salads
        static const int min_radius_hough_salad = 192;
        static const int max_radius_hough_salad = 195;
        
        // Radius parameters to detect plates
        static const int min_radius_refine = 176;
        static const int max_radius_refine = 184;

        // General parameters for hough circles
        static const int param1 = 100;
        static const int param2 = 20;
        static const int ratioMinDist = 2;

        // Parameters to refine salad detection
        static const int paramSalad1 = 150;
        static const int paramSalad2 = 20;
        static const int min_radius_hough_salad_refine = 193;
        static const int max_radius_hough_salad_refine = 202;

    public:

        // Finds plates in a tray
        static std::vector<cv::Vec3f> findPlates(const cv::Mat&);

        // Finds plates the salad bowl in a tray
        static std::vector<cv::Vec3f> findSaladBowl(const cv::Mat&, bool);

        // Finds plates the bread in a tray
        static void findBread(const cv::Mat&, cv::Mat&);
};