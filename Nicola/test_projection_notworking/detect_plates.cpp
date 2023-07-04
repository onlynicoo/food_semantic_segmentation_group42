#include <opencv2/opencv.hpp>
#include <cmath>
#include "../include/PlatesFinder.h"

using namespace cv;

std::vector<Mat> images, images_gray;
Mat src, src_gray, enhancedImage, canny_image, help_img, projected_img, detected_plates;
int imgPerRow = 4;
const char* window_src = "src";
const char* window_enhanced = "enhanced";
const char* window_canny = "canny";
const char* window_lines = "lines";
const char* window_projected = "projected img";
const char* window_plates = "plates img";


//create a class that automatically manage the lines and the various checks

int ratio = 1;
int kernel_size = 7;
int kernel_help = 7;
int lowThreshold = 1000;
const int max_lowThreshold = 1500;
int den = 51; // to set
int hough_parameter = 100; // to set
//280
int min_radius_hough_plates = 280;
int max_radius_hough_plates = 300;
int min_radius_hough_salad = 192;
int max_radius_hough_salad = 195;
int param1 = 100;
int param2 = 20;
int ratioMinDist = 2;



Point lineLineIntersection(Point A, Point B, Point C, Point D)
{
    // Line AB represented as a1x + b1y = c1
    double a1 = B.y - A.y;
    double b1 = A.x - B.x;
    double c1 = a1*(A.x) + b1*(A.y);
 
    // Line CD represented as a2x + b2y = c2
    double a2 = D.y - C.y;
    double b2 = C.x - D.x;
    double c2 = a2*(C.x)+ b2*(C.y);
 
    double determinant = a1*b2 - a2*b1;
 
    if (determinant == 0)
    {
        // The lines are parallel. This is simplified
        // by returning a pair of FLT_MAX
        return Point(FLT_MAX, FLT_MAX);
    }
    else
    {
        double x = (b2*c1 - b1*c2)/determinant;
        double y = (a1*c2 - a2*c1)/determinant;
        return Point(x, y);
    }
}

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

static void CannyThreshold(int, void*) {
    std::cout << "Updating..." << std::endl;

    // Checks that kernel size is always odd
    if( ((kernel_help % 2) == 1) and (kernel_help >= 3) ) kernel_size = kernel_help;

    Size stdSize(0,0);
    Mat imageGrid, imageRow;
    for (int i = 0; i < images.size(); i++) {
        // Get size of the first image, it will be used to display all others
        if (stdSize.empty())
            stdSize = images[i].size();

        Mat output = images[i].clone();
        Mat gray_transformed = images_gray[i].clone();

        // blur and canny the image
        canny_image = enhance(images_gray[i]);

        blur( canny_image, canny_image, Size(21,21) );
        
        Canny( canny_image, canny_image, lowThreshold, lowThreshold*ratio, kernel_size, true );

        //std::cout << "ci siamo\n";

//----------------------------------------------------------------------------------------------------------------------------------------
/*

        // Standard Hough Line Transform
        std::vector<Vec2f> lines;
        HoughLines( canny_image, lines, 1, CV_PI/den, hough_parameter, 0, 0 );

        
        Tray my_tray(canny_image.rows, canny_image.cols);
        
        bool horizontal, vertical;

        for( size_t i = 0; i < lines.size(); i++ ) {
            float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            horizontal = (theta < 0.2 && theta > -0.2);
            vertical = (theta > 1.47 && theta < 1.67);
            if(horizontal || vertical) {
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                my_tray.addLine(vertical, pt1, pt2);
            }
        }

        std::vector<std::vector<Point>> tray = my_tray.getTray();
        if (tray.size() == 4) {

            // Draw the tray
            for (int i = 0; i < 4; i ++) {
                line( output, tray[i][0], tray[i][1], Scalar(0,0,255), 3, LINE_AA);
            }

            // Define points
            std::vector<Point2f> intersection_points;
            std::vector<Point2f> dstPoints;

            // Compute intersection points
            intersection_points.push_back(lineLineIntersection(tray[0][0], tray[0][1], tray[3][0], tray[3][1]));
            circle(output, intersection_points[0], 1, Scalar(0, 255, 0), 100, 8, 0);

            intersection_points.push_back(lineLineIntersection(tray[0][0], tray[0][1], tray[2][0], tray[2][1]));
            circle(output, intersection_points[1], 1, Scalar(0, 255, 0), 100, 8, 0);
            
            intersection_points.push_back(lineLineIntersection(tray[1][0], tray[1][1], tray[2][0], tray[2][1]));
            circle(output, intersection_points[2], 1, Scalar(0, 255, 0), 100, 8, 0);
            
            intersection_points.push_back(lineLineIntersection(tray[1][0], tray[1][1], tray[3][0], tray[3][1]));
            circle(output, intersection_points[3], 1, Scalar(0, 255, 0), 100, 8, 0);

            // add destination points
            dstPoints.push_back(Point(images[i].cols - int(images[i].cols/10), int(images[i].rows/10)));
            circle(output, dstPoints[0], 1, Scalar(0, 0, 255), 100, 8, 0);

            dstPoints.push_back(Point(0+int(images[i].cols/10), 0+int(images[i].rows/10)));
            circle(output, dstPoints[1], 1, Scalar(0, 0, 255), 100, 8, 0);
            
            dstPoints.push_back(Point(0+int(images[i].cols/10), images[i].rows-int(images[i].rows/10)));
            circle(output, dstPoints[2], 1, Scalar(0, 0, 255), 100, 8, 0);
            
            dstPoints.push_back(Point(images[i].cols - int(images[i].cols/10), images[i].rows - int(images[i].rows/10)));
            circle(output, dstPoints[3], 1, Scalar(0, 0, 255), 100, 8, 0);

            cv::Mat transformationMatrix = cv::getPerspectiveTransform(intersection_points, dstPoints);

            //uncomment this to implement the projection
            //cv::warpPerspective(output, output, transformationMatrix, images[i].size());
            //cv::warpPerspective(images_gray[i], gray_transformed, transformationMatrix, images[i].size());
        }
*/
//----------------------------------------------------------------------------------------------------------------------------------------
        //std::cout << "ci siamo\n";

        // Find the circle
        std::vector<Vec3f> circles;
        //std::cout<<"images gray r=" + std::to_string(projected_img.rows) + "images gray c=" + std::to_string(projected_img.cols) + "\n"; 
        HoughCircles(gray_transformed, circles, HOUGH_GRADIENT, 1, gray_transformed.rows / ratioMinDist, param1, param2, min_radius, max_radius);

        //std::cout << "ci siamo\n";

        //std::cout<<"images r=" + std::to_string(images[i].rows) + "images c=" + std::to_string(images[i].cols) + "\n"; 

        // Draw the circle
        for( size_t i = 0; i < circles.size(); i++ ) {
            Vec3i c = circles[i];
            cv::Point center = Point(c[0], c[1]);
            // circle outline
            int radius = c[2];
            circle( output, center, radius, Scalar(255,0,0), 20);
        }

        //std::cout << "ci siamo\n";




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
    imshow(window_plates, imageGrid);
    
    //std::cout << "Done" << std::endl << std::endl;
}
 

// Prints the parameters value
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if  ( event == EVENT_RBUTTONDOWN ) {
    std::cout << "lowThreshold = " << lowThreshold << std::endl;
    std::cout << "ratio = " << ratio << std::endl;
    std::cout << "kernel_size = " << kernel_size << std::endl;
    std::cout << "hough_parameter = " << hough_parameter << std::endl;
    std::cout << "hough_parameter = " << min_radius << std::endl;
    std::cout << "hough_parameter = " << max_radius << std::endl;
    std::cout << "hough_parameter = " << param1 << std::endl;
    std::cout << "hough_parameter = " << param1 << std::endl;
    std::cout << "hough_parameter = " << ratioMinDist << std::endl;
  }
}


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
    
    namedWindow( window_src, WINDOW_AUTOSIZE );

    createTrackbar( "Min Threshold:", window_src, &lowThreshold, max_lowThreshold, CannyThreshold );
    createTrackbar( "Ratio:", window_src, &ratio, 6,  CannyThreshold );
    createTrackbar( "Kernel size:", window_src, &kernel_help, 7, CannyThreshold);
    //createTrackbar( "hough_parameter:", window_src, &hough_parameter, 300, CannyThreshold);
    //createTrackbar( "min_radius:", window_src, &min_radius, 1000, CannyThreshold);
    //createTrackbar( "max_radius:", window_src, &max_radius, 1000, CannyThreshold);
    //setTrackbarMin( "Ratio:", window_src, 1);
    setTrackbarMin( "Kernel size:", window_src, 3);
    createTrackbar("param1:", window_src, &param1, 300, CannyThreshold);
    createTrackbar("param2:", window_src, &param2, 300, CannyThreshold);
    createTrackbar("ratioMinDist:", window_src, &ratioMinDist, 300, CannyThreshold);

    setMouseCallback("Edge Map", CallBackFunc, (void*)NULL);



    namedWindow( window_plates, WINDOW_AUTOSIZE );
    // Display the original and enhanced images
    cv::waitKey(0);

    return 0;
}










//------------------------------------------