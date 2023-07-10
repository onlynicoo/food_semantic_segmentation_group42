
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
int thresholdSaturation = 20;
int thresholdLow = 70;
int thresholdHigh = 120;
int thresholdValueCanny = 122;


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




        /*
        // Apply Canny edge detection
        cv::Mat gray;
        cv::cvtColor(maskedImage, gray, cv::COLOR_BGR2GRAY);
        cv::Mat edges;
        int t1 = 50, t2 = 150;
        cv::Canny(gray, edges, t1, t2);

        //cv::Mat matrix(5, 5, CV_32SC1, cv::Scalar(1));

        cv::Mat kernel = (cv::Mat_<float>(11, 11, 1))/(11*11);
        
        // Apply the sliding kernel using filter2D
        cv::Mat result;
        cv::filter2D(edges, result, -1, kernel);

        // Threshold the result image
        cv::Mat thresholded;
        double thresholdValue = 4*255/11;//90;
        double maxValue = 255;
        cv::threshold(result, thresholded, thresholdValue, maxValue, cv::THRESH_BINARY);


        // Define the structuring element for morphological operation
        cv::Mat kernelclosure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

        // Perform morphological closure
        cv::Mat resultclosure;
        cv::morphologyEx(thresholded, resultclosure, cv::MORPH_CLOSE, kernelclosure);


        cv::Mat labImage;
        cv::cvtColor(maskedImage, labImage, cv::COLOR_BGR2Lab);

        // Split LAB channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Apply histogram equalization to the L channel
        cv::equalizeHist(labChannels[0], labChannels[0]);

        // Merge LAB channels back
        cv::merge(labChannels, labImage);

        // Convert image back to BGR color space
        cv::Mat enhancedImage;
        cv::cvtColor(labImage, enhancedImage, cv::COLOR_Lab2BGR);
        imshow("src nothing", src);
        Mat tmpBlur;
        blur(src, tmpBlur, cv::Size(5,5));
        blur(tmpBlur, tmpBlur, cv::Size(5,5));
        blur(tmpBlur, tmpBlur, cv::Size(5,5));
        imshow("src blur", tmpBlur);
        imshow("tmp enhanced", enhancedImage);
        
        // Convert image to grayscale
        cv::Mat grayscaleImage;
        cv::cvtColor(enhancedImage, grayscaleImage, cv::COLOR_BGR2GRAY);

        // Apply adaptive thresholding
        cv::Mat thresholdedenhanced;
        cv::adaptiveThreshold(grayscaleImage, thresholdedenhanced, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

        // Perform morphological operations to remove small shadows
        cv::Mat kernelenhanced = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(thresholded, thresholdedenhanced, cv::MORPH_OPEN, kernelenhanced);

        // Apply bitwise AND operation to remove shadows from the enhanced image
        cv::Mat shadowRemoved;
        cv::bitwise_and(enhancedImage, enhancedImage, shadowRemoved, thresholdedenhanced);

        */

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

        /* not bad but to improve
            // Convert image to HSV color space
        cv::Mat hsvImage;
        cv::cvtColor(resultlargestComponents, hsvImage, cv::COLOR_BGR2HSV);

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
        
        cv::threshold(saturationChannel, thresholdedSaturation, thresholdSaturation, 255 , cv::THRESH_BINARY);
        cv::threshold(saturationChannel, saturationMask, 1, 255 , cv::THRESH_BINARY);



        // Apply the mask to the original image
        cv::Mat resultsaturation;
        cv::bitwise_and(src, src, resultsaturation, saturationMask-thresholdedSaturation);
        */


       /* bad
        // Convert image from BGR to CIELab color space
        cv::Mat labImage;
        cv::cvtColor(resultlargestComponents, labImage, cv::COLOR_BGR2Lab);

        // Split Lab channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Access and process the L, a, and b channels separately
        cv::Mat lChannel = labChannels[0];
        cv::Mat aChannel = labChannels[1];
        cv::Mat bChannel = labChannels[2];

        // Enhance differences in the a and b channels
        cv::Mat enhancedA, enhancedB;
        cv::equalizeHist(labChannels[1], enhancedA);
        cv::equalizeHist(labChannels[2], enhancedB);
        */




        // Apply Canny edge detection
        cv::Mat gray;
        cv::cvtColor(resultlargestComponents, gray, cv::COLOR_BGR2GRAY);
        cv::Mat edges;
        int t1 = 50, t2 = 150;
        cv::Canny(gray, edges, t1, t2);


        cv::Mat kernelCanny = (cv::Mat_<float>(5, 5, 1))/(5*5);
        
        // Apply the sliding kernel using filter2D
        cv::Mat resultCanny;
        cv::filter2D(edges, result, -1, kernelCanny);

        // Threshold the result image
        cv::Mat thresholdedCanny;
        double maxValueCanny = 255;
        cv::threshold(result, thresholdedCanny, thresholdValueCanny, maxValue, cv::THRESH_BINARY);
        
            // Define the structuring element for closing operation
        int kernelSizeDilation = 3; // Adjust the size according to your needs
        int kernelSizeClosing = 7; // Adjust the size according to your needs
        cv::Mat kernelDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeDilation, kernelSizeDilation));
        cv::Mat kernelClosing = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSizeClosing, kernelSizeClosing));

        // Perform dilation followed by erosion (closing operation)
        cv::Mat closedImage;
        cv::morphologyEx(thresholdedCanny, closedImage, cv::MORPH_DILATE, kernelClosing, cv::Point(1, 1), 3);
        cv::morphologyEx(closedImage, closedImage, cv::MORPH_CLOSE, kernelClosing, cv::Point(1, 1), 3);


        
        // Apply the mask to the original image
        cv::Mat resultDilation;
        cv::bitwise_and(src, src, resultDilation, closedImage);





        cv::Mat output = resultDilation;

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




    createTrackbar( "Min Threshold:", window_plates, &lowThreshold, max_lowThreshold, callBackFunc );
    createTrackbar( "param1:", window_plates, &param1, 300, callBackFunc);
    createTrackbar( "param2:", window_plates, &param2, 300, callBackFunc);
    createTrackbar( "min_radius:", window_plates, &min_radius_hough_plates, 1000, callBackFunc);
    createTrackbar( "max_radius:", window_plates, &max_radius_hough_plates, 1000, callBackFunc);
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
