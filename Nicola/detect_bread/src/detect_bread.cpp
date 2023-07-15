
//#include <opencv2/opencv.hpp>
#include "../include/PlatesFinder.h"

using namespace cv;

Mat output;
const char* window_plates = "src";
std::vector<Mat> images, images_gray;
int imgPerRow = 4;


cv::Mat FindBread(cv::Mat src) {

    // used as base img
    cv::Mat maskedImage = src.clone();

    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    std::vector<cv::Vec3f> salad = PlatesFinder::get_salad(src, true);

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
    // Put this in .h file
    int thresholdLow = 70;
    int thresholdHigh = 117;
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

    // put this in .h file
    int thresholdValueToChange = 111;
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

    // put this in .h file
    int thresholdValueCanny = 115;
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
    //put this in .h file
    int thresholdSaturation = 140;
    cv::threshold(saturationChannel, thresholdedSaturation, thresholdSaturation, 255 , cv::THRESH_BINARY);
    cv::threshold(saturationChannel, saturationMask, 1, 255 , cv::THRESH_BINARY);

    cv::Mat newMask = saturationMask - thresholdedSaturation;

    cv::morphologyEx(newMask, newMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 1);
    cv::morphologyEx(newMask, newMask, cv::MORPH_DILATE, kernelErosion, cv::Point(1, 1), 12);
        
    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(newMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat out = cv::Mat::zeros(newMask.size(), CV_8UC1);

    double thresholdArea = 20000;
    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
        if (area > thresholdArea)
            if (area > largestAreaPost) {
                largestAreaPost = area;
                index = i;
            }    
    }
    if (index != -1) 
        cv::fillPoly(out, contours[index], cv::Scalar(13));

    return out;
}

std::vector<cv::Rect> findBoundingRectangles(const cv::Mat& mask) {
    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find bounding rectangles for each contour
    std::vector<cv::Rect> boundingRectangles;
    for (const auto& contour : contours) {
        cv::Rect boundingRect = cv::boundingRect(contour);
        boundingRectangles.push_back(boundingRect);
    }

    return boundingRectangles;
}

cv::Mat SegmentBread(cv::Mat src, cv::Mat breadMask) {
    cv::Mat image1 = src.clone();
    cv::Mat image = src.clone();
    
    int kernelSizeErosion = 5; // Adjust the size according to your needs
    cv::Mat kernelErosion = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSizeErosion, kernelSizeErosion));
    cv::Mat erodedMask;
    cv::morphologyEx(breadMask, erodedMask, cv::MORPH_ERODE, kernelErosion, cv::Point(1, 1), 3);

    cv::Rect bbx = cv::boundingRect(erodedMask);
	
    Mat final_image, result_mask, bgModel, fgModel;

    // GrabCut segmentation algorithm for current box
    grabCut(image,		// input image
        result_mask,	// segmentation resulting mask
        bbx,			// rectangle containing foreground
        bgModel, fgModel,	// models
        10,				// number of iterations
        cv::GC_INIT_WITH_RECT);
    
    cv::Mat tmpMask0 = (result_mask == 0);
    cv::Mat tmpMask1 = (result_mask == 1);
    cv::Mat tmpMask2 = (result_mask == 2);
    cv::Mat tmpMask3 = (result_mask == 3);

    cv::Mat foreground = (tmpMask3 | tmpMask1);

    // Perform connected component analysis
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);

    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image to hold the filled shapes
    cv::Mat filledMask = cv::Mat::zeros(foreground.size(), CV_8UC1);

    double largestAreaPost = 0;
    int index = -1;
    // Fill the contours of the shapes in the filled mask
    for (int i = 0; i < contours.size(); i ++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestAreaPost) {
            largestAreaPost = area;
            index = i;
        }    
    }
    if (index != -1) 
        cv::fillPoly(filledMask, contours[index], cv::Scalar(13));

    cv::Mat out;
    cv::bitwise_and(src, src, out, filledMask);
    cv::imshow("breadMask", breadMask);
    cv::imshow("out", out);
    cv::waitKey();
    std::cout << "src.size() = " << out.size() << "\n\n\n";

    return filledMask;
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


        cv::Mat tmpMask = SegmentBread(src, FindBread(src));
        cv::Mat tmpImg;
        cv::bitwise_and(src, src, tmpImg, tmpMask);

        cv::Mat output = tmpImg;

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





    createTrackbar( "saturaration:", window_plates, &imgPerRow, 255, callBackFunc);
    setTrackbarMin( "Ratio:", window_plates, 1);
    
    imshow(window_plates, images[1]);
    cv::waitKey();  
    return 0;
}