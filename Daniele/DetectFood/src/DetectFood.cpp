#include <iostream>
#include <opencv2/opencv.hpp>
#include "PlateRemover.h"
#include "FeatureComparator.h"

using namespace std;
using namespace cv;

string LABELS[14] = {
    "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
    "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
    "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

vector<int> excludedLabels = {0, 12, 13};
Mat labelFeatures;
Point center = Point(-1,-1);
int radius = -1;

void onMouse(int event, int x, int y, int f, void* userdata);

int main(int argc, char **argv) {

    // Read arguments
    if (argc < 3) {
        cout << "Usage: food <label features path> <image to label path>" << endl;
        return 1;
    }

    // Read template images
    string labelFeaturesPath = argv[1];
    FileStorage fs(labelFeaturesPath, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Failed to open label features file." << endl;
        return -1;
    }
    fs["labelFeatures"] >> labelFeatures;
    fs.release();

    // Read input image
    string imgPath = argv[2];
    Mat inputImg = imread(imgPath);
    if (inputImg.data == NULL)
        cout << "The given argument is not a valid image" << endl;
    imshow("Input", inputImg);

    // Add mouse callback
    cout << "Click first on plate center and then on plate border..." << endl;
    setMouseCallback("Input", onMouse, (void*)&inputImg);
    waitKey(0);

    return 0;
}

void onMouse(int event, int x, int y, int f, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {

        // Read input image
        Mat img = (*(Mat*)userdata).clone();
        
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
            PlateRemover::getFoodMask(img, mask, center, radius);
            bitwise_and(img, img, masked, mask);

            // Reset params
            center = Point(-1,-1);
            radius = -1;

            // Show output
            imshow("out", masked);

            // Get food label
            Mat imageFeatures = FeatureComparator::getImageFeatures(img, mask);
            int predlabel = FeatureComparator::getFoodLabel(labelFeatures, excludedLabels, imageFeatures);
            cout << "Assigned label: " << LABELS[predlabel] << endl;
            cout << endl;
            
            waitKey();
        }
    }
}