#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const char* imagesWindow = "images";
const char* paramWindow = "params";
vector<Mat> images, grayImages;
int imgPerRow = 4;

// Hough circles parameters
int ratioMinDist = 10;
const int maxRatioMinDist = 25;
int param1 = 100;
const int maxParam1 = 300;
int param2 = 20;
const int maxParam2 = 300;
int minRadius = 250;
const int maxMinRadius = 300;
int maxRadius = 260;
const int maxMaxRadius = 300;

static void HoughCirclesParams(int, void*);

int main(int argc, char** argv) {
    // Read arguments
    if (argc < 2) {
        cout << "You have to pass a dir containing png images as argument" << endl;
        return 1;
    }
    string inputDir = argv[1];

    // Read file names
    vector<String> inputNames;
    glob(inputDir + "/*.jpg", inputNames, true);
    cout << "Input images found: " << inputNames.size() << endl;

    for (int i = 0; i < inputNames.size(); i++) {
        Mat img = imread(inputNames[i], IMREAD_COLOR);
        images.push_back(img);
        cvtColor(img, img, COLOR_BGR2GRAY);
        grayImages.push_back(img);
    }

    // Create windows and set trackbars
    namedWindow(imagesWindow, WINDOW_AUTOSIZE);
    namedWindow(paramWindow, WINDOW_AUTOSIZE);
    createTrackbar("ratioMinDist:", paramWindow, &ratioMinDist, maxRatioMinDist, HoughCirclesParams);
    createTrackbar("param1:", paramWindow, &param1, maxParam1, HoughCirclesParams);
    createTrackbar("param2:", paramWindow, &param2, maxParam2, HoughCirclesParams);
    createTrackbar("minRadius:", paramWindow, &minRadius, maxMinRadius, HoughCirclesParams);
    createTrackbar("maxRadius:", paramWindow, &maxRadius, maxMaxRadius, HoughCirclesParams);
    HoughCirclesParams(0, 0);
    waitKey(0);
    return 0;
}

static void HoughCirclesParams(int, void*) {
    cout << "Updating..." << endl;

    Size stdSize(0,0);
    Mat imageGrid, imageRow;
    for (int i = 0; i < images.size(); i++) {
        // Get size of the first image, it will be used to display all others
        if (stdSize.empty())
            stdSize = images[i].size();

        Mat output;
        // Detect circles using Hough Circle Transform
        vector<Vec3f> circles;
        HoughCircles(grayImages[i], circles, HOUGH_GRADIENT, 1, images[i].rows / ratioMinDist, param1, param2, minRadius, maxRadius);

        // Draw the circles
        output = images[i].clone();
        for (size_t i = 0; i < circles.size(); i++) {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(output, center, 1, Scalar(0, 0, 255), 10, LINE_AA);       // Circle center
            circle(output, center, radius, Scalar(255, 0, 0), 10, LINE_AA);  // Circle outline
        }

        // Resize output to have all images of same size
        resize(output, output, stdSize);
        
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
    resize(imageGrid, imageGrid, stdSize);
    imshow(imagesWindow, imageGrid);
    
    cout << "Done" << endl << endl;
}