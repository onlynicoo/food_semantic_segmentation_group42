#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void preProcessImage(Mat src, Mat &dst) {
    GaussianBlur(src, dst, Size(5,5), 0);
    cvtColor(dst, dst, COLOR_BGR2HSV);
}

void getFoodMask(Mat src, Mat &mask, Point center, int radius) {
    Mat img;
    preProcessImage(src, img);
    mask = Mat(img.rows, img.cols, CV_8U, Scalar(0));

    // Find food mask
    for (int r = max(0, center.y - radius); r < min(center.y + radius + 1, img.rows); r++)
        for (int c = max(0, center.x - radius); c < min(center.x + radius + 1, img.cols); c++) {
            Point cur = Point(c, r);
            if (norm(cur - center) <= radius) {
                // Check if current point is not part of the plate
                int hsv[3] = {int(img.at<Vec3b>(cur)[0]), int(img.at<Vec3b>(cur)[1]), int(img.at<Vec3b>(cur)[2])};
                if (hsv[1] > 80)
                    mask.at<int8_t>(cur) = 255;
            }
        }

    // Fill the holes
    /*int closingSize = radius / 10;  // Adjust the size as per your requirement
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(closingSize, closingSize));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);*/
}