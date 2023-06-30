#include <iostream>
#include <opencv2/opencv.hpp>
#include "plates.h"
#include "FeatureComparator.h"

using namespace std;
using namespace cv;

string LABELS[14] = {
    "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
    "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
    "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

string templateDir;
FeatureComparator comparator;
Point center = Point(-1,-1);
int radius = -1;

void onMouse(int event, int x, int y, int f, void* userdata);

int main(int argc, char **argv) {
    // Read arguments
    if (argc < 3) {
        cout << "Usage: food <template images dir> <image to label path>" << endl;
        return 1;
    }

    // Read template images
    string templateDir = argv[1];
    comparator = FeatureComparator(templateDir);

    // Read input image
    string imgPath = argv[2];
    Mat inputImg = imread(imgPath);
    if (inputImg.data == NULL)
        cout << "The given argument is not a valid image" << endl;
    imshow("Input", inputImg);

    // Add mouse callback
    setMouseCallback("Input", onMouse, (void*)&inputImg);
    waitKey(0);

    return 0;
}

void onMouse(int event, int x, int y, int f, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        // Read input image
        Mat img = (*(Mat*)userdata).clone();

        // Print HSV of the pixel
        /*cvtColor(img, img, COLOR_BGR2HSV);
        int hsv[3] = {int(img.at<Vec3b>(y, x)[0]), int(img.at<Vec3b>(y, x)[1]), int(img.at<Vec3b>(y, x)[2])};
        cout << "HSV at pixel (" << x << "," << y << ") = ";
        cout << "(" << hsv[0] << "," << hsv[1] << "," << hsv[2] << ")\n";*/
        
        if (center.x == -1) {
            // Read center
            center = Point(x, y);
            cout << "Center: " << center << endl;
        } else {
            // Read radius
            radius = norm(Point(x, y) - center);
            cout << "Radius: " << radius << endl;
        }                     
            
        if (center.x != -1 && radius != -1) {

            // Get food mask
            Mat mask, masked;
            getFoodMask(img, mask, center, radius);
            bitwise_and(img, img, masked, mask);

            // Show output
            imshow("out", masked);
            waitKey();

            // Get food label
            int predlabel = comparator.getFoodLabel(img, mask);
            cout << "Assigned label: " << LABELS[predlabel] << endl;

            // Reset params
            center = Point(-1,-1);
            radius = -1;

            cout << endl;
        }
    }
}