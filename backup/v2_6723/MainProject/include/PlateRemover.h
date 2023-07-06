#include <opencv2/opencv.hpp>

class PlateRemover {
    public:
        static void getFoodMask(cv::Mat img, cv::Mat &mask, cv::Point center, int radius);
};