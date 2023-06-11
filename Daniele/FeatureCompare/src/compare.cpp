#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include "features.h"

using namespace std;
using namespace cv;

void preProcessImage(Mat &img, Mat &mask);
int getLabelFromImgPath(string path);
void appendColumns(Mat src, Mat &dst);

int main(int argc, char **argv) {

    // Read arguments
    if (argc < 2) {
        cout << "You have to pass a dir containing png images as argument" << endl;
        return 1;
    }
    string inputDir = argv[1];

    // Read file names
    vector<String> inputNames;
    glob(inputDir + "/*.png", inputNames, false);
    cout << "Input images found: " << inputNames.size() << endl;
    // shuffle(inputNames.begin(), inputNames.end(), random_device());

    // Skip 12 (salad) and 13 (bread) since they should have been already distinguished
    for (int i = 0; i < inputNames.size(); i++)
        if (getLabelFromImgPath(inputNames[i]) >= 12) {
            inputNames.erase(inputNames.begin() + i);
            i--;
        }

    Mat img, mask;
    int numProcessed = 0, numClasses = 13;
    Mat allFeatures;
    vector<Mat> imagesFeatures(numClasses);
    for (int i = 0; i < inputNames.size(); i++) {

        if (numProcessed != 0 && numProcessed % 100 == 0)
            cout << numProcessed << " images processed " << endl;
        
        cout << "Processing image: " << inputNames[i] << endl;

        img = imread(inputNames[i], IMREAD_UNCHANGED);

        preProcessImage(img, mask);

        Mat tmpFeatures;
        appendColumns(0.6 * getHueFeatures(img, mask, 64), tmpFeatures);
        appendColumns(0.4 * getCannyLBPFeatures(img, mask, 64), tmpFeatures);

        Mat* curFeatures = &imagesFeatures[getLabelFromImgPath(inputNames[i]) - 1];

        if (curFeatures->empty())
            tmpFeatures.copyTo(*curFeatures);
        else
            curFeatures->push_back(tmpFeatures);

        if (allFeatures.empty())
            tmpFeatures.copyTo(allFeatures);
        else
            allFeatures.push_back(tmpFeatures);

        /*
        for (int j = 0; j < 10; j++) {
            // Define the size of the crop
            int cropWidth = img.cols / 2;  // Width of the crop
            int cropHeight = img.rows / 2; // Height of the crop

            // Calculate the maximum top-left coordinate for the crop
            int maxLeft = img.cols - cropWidth;
            int maxTop = img.rows - cropHeight;

            // Generate random top-left coordinate for the crop
            int left = rand() % maxLeft;
            int top = rand() % maxTop;

            // Create a rectangle representing the crop
            Rect cropRect(left, top, cropWidth, cropHeight);

            tmpFeatures.release();
            appendColumns(0.6 * getHueFeatures(img(cropRect), mask(cropRect), 64), tmpFeatures);
            appendColumns(0.4 * getCannyLBPFeatures(img(cropRect), mask(cropRect), 64), tmpFeatures);
            if (curFeatures->empty())
                tmpFeatures.copyTo(*curFeatures);
            else
                curFeatures->push_back(tmpFeatures);
        }
        */
        numProcessed++;
    }

    cout << "Total images processed: " << numProcessed << endl;

    cout << "Number of features: " << imagesFeatures[0].cols << endl;

    // Compute average features for every class
    Mat classesFeatures = Mat(numClasses, imagesFeatures[0].cols, CV_32F, Scalar(0));
    for (int i = 0; i < classesFeatures.rows; i++)
        if (!imagesFeatures[i].empty()) {
            reduce(imagesFeatures[i], classesFeatures.row(i), 0, REDUCE_AVG);
        }
    
    int success = 0, total = 0;
    cout << "Wrong predictions: " << endl;
    for (int i = 0; i < inputNames.size(); i++) {
        int label = getLabelFromImgPath(inputNames[i]);
        int predLabel = findNearestCenter(classesFeatures.rowRange(0, numClasses - 2), allFeatures.row(i)) + 1;
        //int predLabel = findNearestCenter(classesFeatures, allFeatures.row(i)) + 1;
        if (label == predLabel)
            success++;
        else
            cout << inputNames[i] << " -> " << predLabel << endl;
        total++;
    }
    cout << "Success rate: " << float(success) / float(total) << endl;
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

void appendColumns(Mat src, Mat &dst) {
    if (dst.empty())
        dst = src;
    else
        hconcat(src, dst, dst);
}