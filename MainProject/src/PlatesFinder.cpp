#include "../include/PlatesFinder.h" // Include the corresponding header file

//Tray::~Tray() {

cv::Mat PlatesFinder::enhance(cv::Mat src){
    // Apply logarithmic transformation
    cv::Mat enhancedImage;
    cv::Mat help = src.clone();
    help.convertTo(enhancedImage, CV_32F); // Convert image to floating-point for logarithmic calculation

    float c = 255.0 / log(1 + 255); // Scaling constant for contrast adjustment
    cv::log(enhancedImage + 1, enhancedImage); // Apply logarithmic transformation
    enhancedImage = enhancedImage * c; // Apply contrast adjustment

    enhancedImage.convertTo(enhancedImage, CV_8U); // Convert back to 8-bit unsigned integer
    return enhancedImage;
}

// Find plate image
std::vector<cv::Vec3f> PlatesFinder::get_plates(cv::Mat src) {

    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
 
    // Find the circle
    std::vector<cv::Vec3f> circles_plate;
    std::vector<cv::Vec3f> circles_salad;
    std::vector<cv::Vec3f> actual_plates;
    std::vector<cv::Vec3f> refine_salad;



    HoughCircles(src_gray, circles_plate, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_plates, max_radius_hough_plates);
    HoughCircles(src_gray, circles_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_hough_salad, max_radius_hough_salad);
    HoughCircles(src_gray, refine_salad, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, param1, param2, min_radius_refine, max_radius_refine);

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

    return actual_plates;

}

// Print plates in image
cv::Mat PlatesFinder::print_plates_image(const cv::Mat src, const std::vector<cv::Vec3f> circles) {
    cv::Mat output = src.clone();
    // Draw the circle
    for( size_t i = 0; i < circles.size(); i++ ) {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle outline
        int radius = c[2];
        circle(output, center, radius, cv::Scalar(255,0,0), 20);
    }
    return output;
}

std::vector<cv::Vec3f> PlatesFinder::get_salad(cv::Mat src, bool saladFound = false) {

    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
 
    // Find the circle
    std::vector<cv::Vec3f> circles_salad;


    HoughCircles(src_gray, circles_salad, cv::HOUGH_GRADIENT, 1, src.rows/ratioMinDist, param1, param2, min_radius_hough_salad, max_radius_hough_salad);
    
    if (circles_salad.size() == 1 || !saladFound) {
            return circles_salad;
    }

    else {
        std::vector<cv::Vec3f> circles_salad_refined;
        HoughCircles(src_gray, circles_salad_refined, cv::HOUGH_GRADIENT, 1, src_gray.rows/ratioMinDist, paramSalad1, paramSalad2, min_radius_hough_salad_refine, max_radius_hough_salad_refine);
        
        std::vector<cv::Vec3f> toRemove = get_plates(src);

        std::vector<cv::Vec3f> actual_plates = circles_salad_refined;

        if (actual_plates.size() > 1) {
            // Remove salad circles
            for (size_t i = 0; i < actual_plates.size(); i++) {
                for (size_t j = 0; j < toRemove.size(); j++) {
                    cv::Vec3i s = actual_plates[i];
                    cv::Vec3i p = toRemove[j];
                    cv::Point center_salad = cv::Point(s[0], s[1]);
                    cv::Point center_toRemove = cv::Point(p[0], p[1]);
                    if (cv::norm(center_toRemove - center_salad) < p[2]) {
                        std::vector<cv::Vec3f>::iterator it = actual_plates.begin() + i;
                        actual_plates.erase(it);
                    }
                }
            }
        }
        return actual_plates;
    }
}