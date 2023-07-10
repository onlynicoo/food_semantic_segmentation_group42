
//#include <opencv2/opencv.hpp>
#include "../include/PlatesFinder.h"

using namespace cv;

Mat output;
const char* window_plates = "src";
std::vector<Mat> images, images_gray;
int imgPerRow = 4;

int ratio = 1;
int kernel_size = 7;
int lowThreshold = 1000;
const int max_lowThreshold = 1500;
int ratioMinDist = 2;
int param1 = 150;
int param2 = 20;
int min_radius_hough_plates = 193;

int max_radius_hough_plates = 202;

/*
int thresholdValueToChange = 111;

int thresholdSaturation = 20;

int thresholdValueCanny = 90;

//int thresholdLow = 83;
int thresholdLow = 70;
int thresholdHigh = 116;
*/

int thresholdValueToChange = 111;
int thresholdSaturation = 160;
int thresholdLow = 70;
int thresholdHigh = 113;
int thresholdValueCanny = 115;


Mat enhance(Mat src){
    // Apply logarithmic transformation
    cv::Mat enhancedImage;
    Mat help = src.clone();
    help.convertTo(enhancedImage, CV_32F); // Convert image to floating-point for logarithmic calculation

    float c = 255.0 / log(1 + 255); // Scaling constant for contrast adjustment
    cv::log(enhancedImage + 1, enhancedImage); // Apply logarithmic transformation
    enhancedImage = enhancedImage * c; // Apply contrast adjustment

    enhancedImage.convertTo(enhancedImage, CV_8U); // Convert back to 8-bit unsigned integer
    return enhancedImage;
}


void callBackFunc(int, void*) {

    Size stdSize(0,0);
    Mat imageGrid, imageRow;
    Mat imageGridSrc, imageRowSrc;
    for (int i = 0; i < images.size(); i++) {
        // Get size of the first image, it will be used to display all others
        if (stdSize.empty())
            stdSize = images[i].size();
        

        Mat src = images[i].clone();


        // it seems good
        //cv::cvtColor(src, YCrCbImage, cv::COLOR_BGR2YUV_YV12);

        std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
        std::vector<cv::Vec3f> salad = PlatesFinder::get_salad(src, true);

        cv::Mat maskedImage = src.clone();

        // Draw the circle
        for( size_t i = 0; i < plates.size(); i++ ) {
            cv::Vec3i c = plates[i];
            cv::Point center = cv::Point(c[0], c[1]);
            // circle outline
            int radius = c[2];
            circle(maskedImage, center, radius*1, cv::Scalar(0,0,0), cv::FILLED);
        }
 
        // Draw the circle
        for( size_t i = 0; i < salad.size(); i++ ) {
            cv::Vec3i c = salad[i];
            cv::Point center = cv::Point(c[0], c[1]);
            // circle outline
            int radius = c[2];
            circle(maskedImage, center, radius*1.4, cv::Scalar(0,0,0), cv::FILLED);
        }

        // Convert image to YUV color space
        cv::Mat yuvImage;
        cv::cvtColor(maskedImage, yuvImage, cv::COLOR_BGR2YUV);

        // Access and process the Y, U, and V channels separately
        std::vector<cv::Mat> yuvChannels;
        cv::split(yuvImage, yuvChannels);

        // Create a binary mask of pixels within the specified range
        cv::Mat mask;
        cv::inRange(yuvChannels[1], thresholdLow, thresholdHigh, mask);

        // Apply the mask to the original image
        cv::Mat resultyuv;
        cv::bitwise_and(yuvChannels[1], yuvChannels[1], resultyuv, mask);


        // Define the structuring element for morphological operation
        cv::Mat kernelclosure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

        // Perform morphological closure
        cv::Mat resultclosure;
        cv::morphologyEx(resultyuv, resultclosure, cv::MORPH_CLOSE, kernelclosure);





        // Perform connected component analysis
        cv::Mat labels, stats, centroids;
        int numComponents = cv::connectedComponentsWithStats(resultclosure, labels, stats, centroids);

        // Find the connected component with the largest area
        int largestComponent = 0;
        int largestArea = 0;
        for (int i = 1; i < numComponents; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > largestArea) {
                largestArea = area;
                largestComponent = i;
            }
        }

        // Create a binary mask for the largest component
        cv::Mat largestComponentMask = (labels == largestComponent);


        cv::Mat kernel = (cv::Mat_<float>(51, 51, 1))/(45*45);
        
        // Apply the sliding kernel using filter2D
        cv::Mat result;
        cv::filter2D(largestComponentMask, result, -1, kernel);

        // Threshold the result image
        cv::Mat thresholdedLargestComponentMask;
        double maxValue = 255;
        cv::threshold(result, thresholdedLargestComponentMask, thresholdValueToChange, maxValue, cv::THRESH_BINARY);



        
        // Apply the mask to the original image
        cv::Mat resultlargestComponents;
        cv::bitwise_and(src, src, resultlargestComponents, thresholdedLargestComponentMask);



        // Apply Canny edge detection
        cv::Mat gray;
        cv::cvtColor(resultlargestComponents, gray, cv::COLOR_BGR2GRAY);
        cv::Mat edges;
        int t1 = 50, t2 = 150;
        cv::Canny(gray, edges, t1, t2);


        cv::Mat kernelCanny = (cv::Mat_<float>(7, 7, 1))/(7*7);
        
        // Apply the sliding kernel using filter2D
        cv::Mat resultCanny;
        cv::filter2D(edges, result, -1, kernelCanny);

        // Threshold the result image
        cv::Mat thresholdedCanny;
        double maxValueCanny = 255;
        cv::threshold(result, thresholdedCanny, thresholdValueCanny, maxValue, cv::THRESH_BINARY);
        
        // Define the structuring element for closing operation
        int kernelSizeDilation = 3; // Adjust the size according to your needs
        int kernelSizeClosing = 5; // Adjust the size according to your needs
        int kernelSizeErosion = 3; // Adjust the size according to your needs
        cv::Mat kernelDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeDilation, kernelSizeDilation));
        cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeClosing, kernelSizeClosing));
        cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeErosion, kernelSizeErosion));

        // Perform dilation followed by erosion (closing operation)
        cv::Mat closedImage;
        cv::morphologyEx(thresholdedCanny, closedImage, cv::MORPH_DILATE, kernelDilation, cv::Point(1, 1), 10);
        cv::morphologyEx(closedImage, closedImage, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 4);

        


        cv::morphologyEx(closedImage, closedImage, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 6);

        cv::morphologyEx(closedImage, closedImage, cv::MORPH_DILATE, kernelDilation, cv::Point(1, 1), 20);


        // Apply the mask to the original image
        cv::Mat res;
        cv::bitwise_and(src, src, res, closedImage);


        //not bad but to improve
        // Convert image to HSV color space
        cv::Mat hsvImage;
        cv::cvtColor(res, hsvImage, cv::COLOR_BGR2HSV);

            // Split HSV channels
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsvImage, hsvChannels);

        // Access and process the H, S, and V channels separately
        cv::Mat hueChannel = hsvChannels[0];
        cv::Mat saturationChannel = hsvChannels[1];
        cv::Mat valueChannel = hsvChannels[2];


        // Threshold the result image
        cv::Mat thresholdedSaturation;
        cv::Mat saturationMask;
        thresholdSaturation = 140;
        cv::threshold(saturationChannel, thresholdedSaturation, thresholdSaturation, 255 , cv::THRESH_BINARY);
        cv::threshold(saturationChannel, saturationMask, 1, 255 , cv::THRESH_BINARY);

        cv::Mat newMask = saturationMask - thresholdedSaturation;

        cv::morphologyEx(newMask, newMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 1);
        cv::morphologyEx(newMask, newMask, cv::MORPH_DILATE, kernelErosion, cv::Point(1, 1), 6);
        


        // Apply the mask to the original image
        cv::Mat resultsaturation;
        
        cv::bitwise_and(src, src, resultsaturation, newMask);
        

        
        



        cv::Mat output = resultsaturation;

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
    cv::imshow(window_plates, imageGrid);   
    cv::waitKey();


}


// Prints the parameters value
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if  ( event == EVENT_RBUTTONDOWN ) {
    std::cout << "thresholdHigh = " << thresholdHigh << std::endl;
    std::cout << "thresholdLow = " << thresholdLow << std::endl;
    std::cout << "thresholdValueToChange = " << thresholdValueToChange << std::endl;
    std::cout << "thresholdValueCanny = " << thresholdValueCanny << std::endl;
  }
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





    createTrackbar( "saturaration:", window_plates, &thresholdSaturation, 255, callBackFunc);
    createTrackbar( "canny:", window_plates, &thresholdValueCanny, 255, callBackFunc);
    createTrackbar( "thresholdLow:", window_plates, &thresholdLow, 255, callBackFunc);
    createTrackbar( "thresholdHigh  :", window_plates, &thresholdHigh, 255, callBackFunc);
    createTrackbar( "thresholdValueToChange  :", window_plates, &thresholdValueToChange, 255, callBackFunc);
    setTrackbarMin( "Ratio:", window_plates, 1);

    setMouseCallback(window_plates, CallBackFunc, (void*)NULL);
    
    imshow(window_plates, images[1]);
    cv::waitKey();  
    return 0;
}