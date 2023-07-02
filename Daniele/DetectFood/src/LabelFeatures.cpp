#include <iostream>
#include <opencv2/opencv.hpp>
#include "FeatureComparator.h"

using namespace std;
using namespace cv;

const int N_LABELS = 14;

int getLabelFromImgPath(string path);
void preProcessImage(Mat &img, Mat &mask);

int main(int argc, char **argv) {

    // Read arguments
    if (argc < 2) {
        cout << "You have to pass a dir containing .png images as argument" << endl;
        return 1;
    }
    string inputDir = argv[1];

    // Read file names
    vector<String> inputNames;
    glob(inputDir + "/*.png", inputNames, false);
    cout << "Input images found: " << inputNames.size() << endl;

    int numProcessed = 0, numFeatures = -1;
    vector<Mat> imagesFeatures(N_LABELS);
    for (int i = 0; i < inputNames.size(); i++) {

        Mat img, mask;
        img = imread(inputNames[i], IMREAD_UNCHANGED);
        int label = getLabelFromImgPath(inputNames[i]);

        preProcessImage(img, mask);
        Mat features = FeatureComparator::getImageFeatures(img, mask);

        if (numFeatures == -1)
            numFeatures = features.cols;

        Mat* curFeatures = &imagesFeatures[label];
        if (curFeatures->empty())
            features.copyTo(*curFeatures);
        else
            curFeatures->push_back(features);

        numProcessed++;
    }

    cout << "Total images processed: " << numProcessed << endl;
    cout << "Number of features: " << numFeatures << endl;

    // Compute average features for every class
    Mat labelFeatures = Mat(N_LABELS, numFeatures, CV_32F, Scalar(0));
    for (int i = 0; i < labelFeatures.rows; i++)
        if (!imagesFeatures[i].empty()) {
            reduce(imagesFeatures[i], labelFeatures.row(i), 0, REDUCE_AVG);
        }

    FileStorage fs("label_features.yml", FileStorage::WRITE);
    fs << "labelFeatures" << labelFeatures;
    fs.release();
}

void preProcessImage(Mat &img, Mat &mask) {
    // Find bounding box
    int minHor = INT_MAX, maxHor = INT_MIN, minVer = INT_MAX, maxVer = INT_MIN;
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++) {
            auto& pixel = img.at<Vec4b>(r,c);
            int alpha = int(pixel[3]);
            if (alpha == 255) {
                if (r < minVer)
                    minVer = r;
                if (r > maxVer)
                    maxVer = r;
                if (c < minHor)
                    minHor = c;
                if (c > maxHor)
                    maxHor = c;
            }
        }
    
    // Crop image to bounding box
    img = img(Rect(minHor, minVer, maxHor - minHor, maxVer - minVer));

    // Find mask
    mask = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++) {
            auto& pixel = img.at<Vec4b>(r,c);
            int alpha = int(pixel[3]);
            if (alpha == 255) {
                mask.at<int8_t>(r,c) = 255;
            }
        }
}

int getLabelFromImgPath(string path) {
    string name = path.substr(path.find_last_of('\\') + 1, path.find_last_of(".") - 1 - path.find_last_of('\\'));
    int label = stoi(name.substr(2, name.find_first_of('_') - 2));
    return label;
}