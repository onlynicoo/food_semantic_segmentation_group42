//#ifndef PLATES_FINDER // Header guard to prevent multiple inclusions
//#define PLATES_FINDER

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

    // ceneral parameters for hough circles
    static const int param1 = 100;
    static const int param2 = 20;
    static const int ratioMinDist = 2;

    // log stratch to enhance the contrast
    static cv::Mat enhance(cv::Mat);

  public:

    // Find plate image
    static std::vector<cv::Vec3f> get_plates(cv::Mat);

    // Print plates in image
    static cv::Mat print_plates_image(const cv::Mat, const std::vector<cv::Vec3f>);
};

//#endif // End of header guard