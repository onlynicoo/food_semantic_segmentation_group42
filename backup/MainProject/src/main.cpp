#include <opencv2/opencv.hpp>
#include "../include/Tray.h"

using namespace cv;

Mat src;
const char* window_plates = "src";

int main( int argc, char** argv )
{
    // read the image
    std::string before = argv[1];
    std::string after = argv[2];
    
    Tray my_tray = Tray(before, after);
    
    my_tray.PrintInfo(); 

    //cv::waitKey(0);

    return 0;
}
