#pragma once
#include <opencv2/opencv.hpp>

// Declaration of the class PlateFinder
class PlatesFinder {

  private:
    // Attributes

    // radius parameters to detect plates
    static const int min_radius_hough_plates = 280;
    static const int max_radius_hough_plates = 300;
    
    // radius parameters to detect salads
    static const int min_radius_hough_salad = 192;
    static const int max_radius_hough_salad = 195;
    
    // radius parameters to detect plates
    static const int min_radius_refine = 176;
    static const int max_radius_refine = 184;

    // general parameters for hough circles
    static const int param1 = 100;
    static const int param2 = 20;
    static const int ratioMinDist = 2;

    // parameters to refine salad detection
    static const int paramSalad1 = 150;
    static const int paramSalad2 = 20;
    static const int min_radius_hough_salad_refine = 193;
    static const int max_radius_hough_salad_refine = 202;


    // log stratch to enhance the contrast
    static cv::Mat enhance(cv::Mat);

  public:

    // Find plate in image
    static std::vector<cv::Vec3f> get_plates(cv::Mat);

    // Find salads in image
    static std::vector<cv::Vec3f> get_salad(cv::Mat, bool);

    // Print plates in image
    static cv::Mat print_plates_image(const cv::Mat, const std::vector<cv::Vec3f>);

};