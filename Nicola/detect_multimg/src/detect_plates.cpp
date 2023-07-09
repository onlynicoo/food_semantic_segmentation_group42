
//#include <opencv2/opencv.hpp>
#include "../include/PlatesFinder.h"

using namespace cv;

Mat output;
const char* window_plates = "src";
std::vector<Mat> images, images_gray;
int imgPerRow = 3;

int lowThreshold = 1000;
const int max_lowThreshold = 1500;
int ratioMinDist = 2;
int param1 = 150;
int param2 = 20;
int min_radius_hough_plates = 193;
int max_radius_hough_plates = 202;



void callBackFunc(int, void*) {

    Size stdSize(0,0);
    Mat imageGrid, imageRow;
    for (int i = 0; i < images.size(); i++) {
        // Get size of the first image, it will be used to display all others
        if (stdSize.empty())
            stdSize = images[i].size();
        

        Mat src = images[i].clone();




        std::vector<cv::Vec3f> circles = PlatesFinder::get_plates(src);

        cv::Mat src_gray;
        cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
 
        // Find the circle
        std::vector<cv::Vec3f> circles_plate;
        
        HoughCircles(src_gray, circles_plate, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_plates, max_radius_hough_plates);
        
        output = PlatesFinder::print_plates_image(src, circles_plate);

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
    

    cv::waitKey();

}





 



int main( int argc, char** argv )
{

    namedWindow( window_plates, WINDOW_AUTOSIZE );
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




    createTrackbar( "Min Threshold:", window_plates, &lowThreshold, max_lowThreshold, callBackFunc );
    createTrackbar( "param1:", window_plates, &param1, 300, callBackFunc);
    createTrackbar( "param2:", window_plates, &param2, 300, callBackFunc);
    createTrackbar( "min_radius:", window_plates, &min_radius_hough_plates, 1000, callBackFunc);
    createTrackbar( "max_radius:", window_plates, &max_radius_hough_plates, 1000, callBackFunc);
    setTrackbarMin( "Ratio:", window_plates, 1);

    imshow(window_plates, images[1]);
    cv::waitKey();  
    return 0;
}
