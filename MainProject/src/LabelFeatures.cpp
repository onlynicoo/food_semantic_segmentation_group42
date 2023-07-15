#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/FeatureComparator.h"

const int N_LABELS = 14;
const int N_TRAYS = 8;
const char* IMG_EXT = ".jpg";
const char* MASK_EXT = ".png";
std::vector<std::string> trayNames = {"food_image", "leftover1", "leftover2", "leftover3"};

int main(int argc, char** argv) {
    std::string inputDir = "../input/Food_leftover_dataset";

    std::cout << "Processing images in " + inputDir << std::endl;

    int numProcessed = 0, numFeatures = -1;
    std::vector<cv::Mat> imagesFeatures(N_LABELS);
    for (int i = 0; i < N_TRAYS; i++)
        for (int j = 0; j < trayNames.size(); j++) {
            // Read image and mask
            std::string imgName = inputDir + "/tray" + std::to_string(i + 1) + "/" + trayNames[j] + IMG_EXT;
            std::string maskName = inputDir + "/tray" + std::to_string(i + 1) + "/masks/" + trayNames[j];
            if (j == 0)
                maskName += "_mask";
            maskName += MASK_EXT;

            cv::Mat img = cv::imread(imgName, cv::IMREAD_COLOR), mask = imread(maskName, cv::IMREAD_GRAYSCALE);

            // Read each masked label
            for (int label = 1; label < N_LABELS; label++) {
                cv::Mat labelMask;
                compare(mask, label, labelMask, cv::CMP_EQ);

                if (cv::countNonZero(labelMask) == 0)
                    continue;

                // If not empty, compute features for the patch
                cv::Mat features;
                FeatureComparator::getImageFeatures(img, labelMask, features);

                if (numFeatures == -1)
                    numFeatures = features.cols;

                // Add features
                cv::Mat* curFeatures = &imagesFeatures[label];
                if (curFeatures->empty())
                    features.copyTo(*curFeatures);
                else
                    curFeatures->push_back(features);

                numProcessed++;
            }
        }

    std::cout << "Total processed patches: " << numProcessed << std::endl;
    std::cout << "Number of features: " << numFeatures << std::endl;

    // Compute average features for every label
    cv::Mat labelFeatures = cv::Mat(N_LABELS, numFeatures, CV_32F, cv::Scalar(0));
    for (int i = 0; i < labelFeatures.rows; i++)
        if (!imagesFeatures[i].empty()) {
            reduce(imagesFeatures[i], labelFeatures.row(i), 0, cv::REDUCE_AVG);
        }

    FeatureComparator::writeLabelFeaturesToFile(labelFeatures);
}