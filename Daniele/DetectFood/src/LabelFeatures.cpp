#include <iostream>
#include <opencv2/opencv.hpp>
#include "FeatureComparator.h"

using namespace std;
using namespace cv;

const int N_LABELS = 14;
const int N_TRAYS = 8;
const char* IMG_EXT = ".jpg";
const char* MASK_EXT = ".png";
vector<string> trayNames = {"food_image", "leftover1", "leftover2", "leftover3"};

int main(int argc, char **argv) {

    // Read arguments
    if (argc < 2) {
        cout << "You have to pass a the Food_leftover_dataset dir as argument" << endl;
        return 1;
    }
    string inputDir = argv[1];

    cout << "Processing images in " + inputDir << endl;

    int numProcessed = 0, numFeatures = -1;
    vector<Mat> imagesFeatures(N_LABELS);
    for (int i = 0; i < N_TRAYS; i++)
        for (int j = 0; j < trayNames.size(); j++) {

            // Read image and mask
            string imgName = inputDir + "/tray" + to_string(i + 1) + "/" + trayNames[j] + IMG_EXT;
            string maskName = inputDir + "/tray" + to_string(i + 1) + "/masks/" + trayNames[j];
            if (j == 0)
                maskName += "_mask";
            maskName += MASK_EXT;

            Mat img = imread(imgName, IMREAD_COLOR), mask = imread(maskName, IMREAD_GRAYSCALE);

            // Read each masked label
            for (int label = 1; label < N_LABELS; label++) {

                Mat labelMask;
                compare(mask, label, labelMask, CMP_EQ);

                if (countNonZero(labelMask) == 0)
                    continue;

                // If not empty, compute features for the patch
                Mat features = FeatureComparator::getImageFeatures(img, labelMask);

                if (numFeatures == -1)
                    numFeatures = features.cols;

                // Add features
                Mat* curFeatures = &imagesFeatures[label];
                if (curFeatures->empty())
                    features.copyTo(*curFeatures);
                else
                    curFeatures->push_back(features);

                numProcessed++;
            }
        }

    cout << "Total processed patches: " << numProcessed << endl;
    cout << "Number of features: " << numFeatures << endl;

    // Compute average features for every label
    Mat labelFeatures = Mat(N_LABELS, numFeatures, CV_32F, Scalar(0));
    for (int i = 0; i < labelFeatures.rows; i++)
        if (!imagesFeatures[i].empty()) {
            reduce(imagesFeatures[i], labelFeatures.row(i), 0, REDUCE_AVG);
        }

    FileStorage fs("label_features.yml", FileStorage::WRITE);
    fs << "labelFeatures" << labelFeatures;
    fs.release();
}