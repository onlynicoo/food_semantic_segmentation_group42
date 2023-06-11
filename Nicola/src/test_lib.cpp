#include <opencv2/opencv.hpp>
#include <cmath>
#include "../include/Tray.h"

using namespace cv;

Mat src, src_gray, enhancedImage, detected_edges, detected_lines, help_img, projected_img, detected_plates;
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
const int max_lowThreshold = 1000;
int den = 51; // to set
int hough_parameter = 100; // to set
int min_radius = 300;
int max_radius = 600;

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
    // Checks that kernel size is always odd
    if( ((kernel_help % 2) == 1) and (kernel_help >= 3) ) kernel_size = kernel_help;

    // blur and canny the image
    detected_edges = enhance(src_gray);
    blur( detected_edges, detected_edges, Size(5,5) );
    
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    // Standard Hough Line Transform
    std::vector<Vec2f> lines; // will hold the results of the detection
    HoughLines( detected_edges, lines, 1, CV_PI/den, hough_parameter, 0, 0 ); // runs the actual detection

    // Find the points of the lines
    //Point points[4];
    detected_lines = help_img.clone();

    //std::cout << detected_lines.rows << std::endl;

    Tray my_tray(detected_lines.rows, detected_lines.cols);

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
        for (int i = 0; i < 4; i ++) {
            line( detected_lines, tray[i][0], tray[i][1], Scalar(0,0,255), 3, LINE_AA);
            //imshow( window_lines, detected_lines);
            //waitKey();
        }

        std::vector<Point2f> intersection_points;
        std::vector<Point2f> dstPoints;

        intersection_points.push_back(lineLineIntersection(tray[0][0], tray[0][1], tray[3][0], tray[3][1]));
        circle(detected_lines, intersection_points[0], 1, Scalar(0, 255, 0), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();

        intersection_points.push_back(lineLineIntersection(tray[0][0], tray[0][1], tray[2][0], tray[2][1]));
        circle(detected_lines, intersection_points[1], 1, Scalar(0, 255, 0), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();
        
        intersection_points.push_back(lineLineIntersection(tray[1][0], tray[1][1], tray[2][0], tray[2][1]));
        circle(detected_lines, intersection_points[2], 1, Scalar(0, 255, 0), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();
        
        intersection_points.push_back(lineLineIntersection(tray[1][0], tray[1][1], tray[3][0], tray[3][1]));
        circle(detected_lines, intersection_points[3], 1, Scalar(0, 255, 0), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();
        

        dstPoints.push_back(Point(detected_lines.cols - int(detected_lines.cols/10), int(detected_lines.rows/10)));

        circle(detected_lines, dstPoints[0], 1, Scalar(0, 0, 255), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();
        
        dstPoints.push_back(Point(0+int(detected_lines.cols/10), 0+int(detected_lines.rows/10)));
        circle(detected_lines, dstPoints[1], 1, Scalar(0, 0, 255), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();

        dstPoints.push_back(Point(0+int(detected_lines.cols/10), detected_lines.rows-int(detected_lines.rows/10)));
        circle(detected_lines, dstPoints[2], 1, Scalar(0, 0, 255), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();

        dstPoints.push_back(Point(detected_lines.cols - int(detected_lines.cols/10), detected_lines.rows - int(detected_lines.rows/10)));
        circle(detected_lines, dstPoints[3], 1, Scalar(0, 0, 255), 100, 8, 0);
        //imshow( window_lines, detected_lines);
        //waitKey();

        cv::Mat transformationMatrix = cv::getPerspectiveTransform(intersection_points, dstPoints);

        
        cv::warpPerspective(detected_lines, projected_img, transformationMatrix, detected_lines.size());
    }

      // Find the circle
    std::vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1,
                    src_gray.rows/16,
                    100, 30, 
                    min_radius, 
                    max_radius 
    );

    detected_plates = projected_img.clone();

    // Draw the circle
    for( size_t i = 0; i < circles.size(); i++ ) {
        Vec3i c = circles[i];
        cv::Point center = Point(c[0], c[1]);
        // circle outline
        int radius = c[2];
        circle( detected_plates, center, radius, Scalar(255,0,0), 2);
    }

    imshow( window_plates, detected_plates);
    imshow( window_projected, projected_img);
    imshow( window_canny, detected_edges);
    imshow( window_lines, detected_lines);
}



int main( int argc, char** argv )
{
    // Read the input image
    Mat src = cv::imread(argv[1], IMREAD_COLOR);
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    detected_lines = src.clone();
    help_img = src.clone();
    namedWindow( window_src, WINDOW_AUTOSIZE );
    namedWindow( window_canny, WINDOW_AUTOSIZE );
    namedWindow( window_lines, WINDOW_AUTOSIZE );
    namedWindow( window_projected, WINDOW_AUTOSIZE );
    namedWindow( window_plates, WINDOW_AUTOSIZE );
    enhancedImage = enhance(src);
    
    createTrackbar( "Min Threshold:", window_src, &lowThreshold, max_lowThreshold, CannyThreshold );
    createTrackbar( "Ratio:", window_src, &ratio, 6,  CannyThreshold );
    createTrackbar( "Kernel size:", window_src, &kernel_help, 7, CannyThreshold);
    createTrackbar( "hough_parameter:", window_src, &hough_parameter, 7000, CannyThreshold);
    createTrackbar( "min_radius:", window_src, &min_radius, 1000, CannyThreshold);
    createTrackbar( "max_radius:", window_src, &max_radius, 1000, CannyThreshold);
    setTrackbarMin( "Ratio:", window_src, 1);
    setTrackbarMin( "Kernel size:", window_src, 3);
    // Display the original and enhanced images
    cv::imshow(window_src, src);
    cv::imshow(window_enhanced, enhancedImage);
    cv::waitKey(0);

    return 0;
}


//------------------------------------------



