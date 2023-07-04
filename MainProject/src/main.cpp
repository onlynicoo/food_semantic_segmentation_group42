#include <opencv2/opencv.hpp>
#include "../include/Trays.h"

using namespace cv;

Mat src;
const char* window_plates = "src";

int main( int argc, char** argv )
{
    // read the image
    std::string before = argv[1];
    std::string after = argv[2];
    
    Trays my_trays = Trays(before, after);
    
    my_trays.PrintInfo(); 

    my_trays.AddTray(before, after); 

    my_trays.PrintInfo(); 

    //cv::waitKey(0);

    return 0;
}
