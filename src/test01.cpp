#include <opencv2/opencv.hpp>
#include <cmath>
using namespace cv;

Mat src, src_gray, enhancedImage, detected_edges, detected_lines, help_img;
const char* window_src = "src";
const char* window_enhanced = "enhanced";
const char* window_canny = "canny";
const char* window_lines = "lines";




int ratio = 3;
int kernel_size = 3;
int kernel_help = 3;
int lowThreshold = 0;
const int max_lowThreshold = 1000;
int den = 51; // to set
int hough_parameter = 100; // to set


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

    std::cout << detected_lines.rows << std::endl;

    for( size_t i = 0; i < lines.size(); i++ ) {
        float rho = lines[i][0], theta = lines[i][1];
        std::cout<<"theta: "<<theta<<std::endl;
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        if((theta < 0.2 && theta > -0.2) || (theta > 1.47 && theta < 1.67)) {
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( detected_lines, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);

        }
    }
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
    std::cout << src.rows << std::endl;
    std::cout << detected_lines.rows << std::endl;
    namedWindow( window_src, WINDOW_AUTOSIZE );
    namedWindow( window_canny, WINDOW_AUTOSIZE );
    namedWindow( window_lines, WINDOW_AUTOSIZE );
    enhancedImage = enhance(src);
    
    createTrackbar( "Min Threshold:", window_src, &lowThreshold, max_lowThreshold, CannyThreshold );
    createTrackbar( "Ratio:", window_src, &ratio, 6,  CannyThreshold );
    createTrackbar( "Kernel size:", window_src, &kernel_help, 7, CannyThreshold);
    createTrackbar( "hough_parameter:", window_src, &hough_parameter, 7000, CannyThreshold);
    setTrackbarMin( "Ratio:", window_src, 1);
    setTrackbarMin( "Kernel size:", window_src, 3);
    // Display the original and enhanced images
    cv::imshow(window_src, src);
    cv::imshow(window_enhanced, enhancedImage);
    cv::waitKey(0);

    return 0;
}


//------------------------------------------