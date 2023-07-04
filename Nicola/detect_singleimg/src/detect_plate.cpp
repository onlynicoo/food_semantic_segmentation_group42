
//#include <opencv2/opencv.hpp>
#include "../include/PlatesFinder.h"

using namespace cv;

Mat src;
const char* window_plates = "src";

int main( int argc, char** argv )
{
    // read the image
    Mat src = imread(argv[1], IMREAD_COLOR);
    
    std::vector<cv::Vec3f> circles = PlatesFinder::get_plates(src);
    
    imshow(window_plates, PlatesFinder::print_plates_image(src, circles));

    cv::waitKey(0);

    return 0;
}
