#include <opencv2/opencv.hpp>
#include <cmath>


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
static const int min_radius_refine = 176;
static const int max_radius_refine = 184;



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

        // Find the circle
        std::vector<cv::Vec3f> circles_plate;
        std::vector<cv::Vec3f> circles_salad;
        std::vector<cv::Vec3f> actual_plates;
        std::vector<cv::Vec3f> refine_salad;



        HoughCircles(canny_image, circles_plate, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_plates, max_radius_hough_plates);
        HoughCircles(canny_image, circles_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_salad, max_radius_hough_salad);
        HoughCircles(canny_image, refine_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_refine, max_radius_refine);

        actual_plates = circles_plate;

        // Remove salad circles
        for (size_t i = 0; i < circles_salad.size(); i++) {
            for (size_t j = 0; j < circles_plate.size(); j++) {
                cv::Vec3i s = circles_salad[i];
                cv::Vec3i p = circles_plate[j];
                cv::Point center_salad = cv::Point(s[0], s[1]);
                cv::Point center_plate = cv::Point(p[0], p[1]);
                if (cv::norm(center_plate - center_salad) < p[2]) {
                    std::vector<cv::Vec3f>::iterator it = actual_plates.begin()+j;
                    actual_plates.erase(it);
                }
            }
        }

        if (actual_plates.size() > 2) {
            // Remove salad circles
            for (size_t i = 0; i < actual_plates.size(); i++) {
                for (size_t j = 0; j < refine_salad.size(); j++) {
                    cv::Vec3i s = actual_plates[i];
                    cv::Vec3i p = refine_salad[j];
                    cv::Point center_salad = cv::Point(s[0], s[1]);
                    cv::Point center_plate = cv::Point(p[0], p[1]);
                    if (cv::norm(center_plate - center_salad) < p[2]) {
                        std::vector<cv::Vec3f>::iterator it = actual_plates.begin() + i;
                        actual_plates.erase(it);
                    }
                }
            }
        }


        // Draw the circle
        for( size_t i = 0; i < actual_plates.size(); i++ ) {
            Vec3i c = actual_plates[i];
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
}
 

// Prints the parameters value
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if  ( event == EVENT_RBUTTONDOWN ) {
    std::cout << "lowThreshold = " << lowThreshold << std::endl;
    std::cout << "ratio = " << ratio << std::endl;
    std::cout << "kernel_size = " << kernel_size << std::endl;
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