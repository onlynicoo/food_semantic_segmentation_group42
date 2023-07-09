
//#include <opencv2/opencv.hpp>
#include "../include/PlatesFinder.h"

using namespace cv;

Mat output;
const char* window_plates = "src";
std::vector<Mat> images, images_gray;
int imgPerRow = 4;

int main( int argc, char** argv )
{
    // Read arguments
    if (argc < 2) {
        std::cout << "You have to pass a dir containing png images as argument" << std::endl;
        return 1;
    }
    std::string inputDir = argv[1];

    // Read file names
    std::vector<String> inputNames;
    glob(inputDir + "/*.jpg", inputNames, true);

    std::cout << "Input images found: " << inputNames.size() << std::endl;

    for (int i = 0; i < inputNames.size(); i++) {
        Mat img = imread(inputNames[i], IMREAD_COLOR);
        images.push_back(img);
        cvtColor(img, img, COLOR_BGR2GRAY);
        images_gray.push_back(img);    
    }

    namedWindow( window_plates, WINDOW_AUTOSIZE );


    Size stdSize(0,0);
    Mat imageGrid, imageRow;
    for (int i = 0; i < images.size(); i++) {
        // Get size of the first image, it will be used to display all others
        if (stdSize.empty())
            stdSize = images[i].size();
        
        Mat src = images[i].clone();
        std::vector<cv::Vec3f> circles = PlatesFinder::get_plates(src);
        output = PlatesFinder::print_plates_image(src, circles);

        std::cout << "rows: " << src.rows << " ";
        std::cout << "cols: " << src.cols << "\n";
        
        // Resize output to have all images of same size
        cv::resize(output, output, stdSize);
        
                // Add image to current image row
        if (imageRow.empty())
            output.copyTo(imageRow);
        else
            hconcat(output, imageRow, imageRow);

        // If the row is full add row to image grid and start a new row
        if ((i + 1) % imgPerRow == 0) {
            if (imageGrid.empty())
                imageRow.copyTo(imageGrid);
            else
                vconcat(imageRow, imageGrid, imageGrid);
            imageRow.release();
        }
    }
    // Resize the full image grid and display it
    cv::resize(imageGrid, imageGrid, stdSize);
    cv::imshow(window_plates, imageGrid);
    

    cv::waitKey(0);

    return 0;
}
