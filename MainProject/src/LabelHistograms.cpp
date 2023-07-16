#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/HistogramComparator.h"

const int N_LABELS = 14;
const int N_TRAYS = 8;
const char* IMG_EXT = ".jpg";
const char* MASK_EXT = ".png";
std::vector<std::string> trayNames = {"food_image", "leftover1", "leftover2", "leftover3"};

int main(int argc, char** argv) {
    std::string inputDir = "../input/Food_leftover_dataset";

    std::cout << "Processing images in " + inputDir << std::endl;

    int numProcessed = 0, numHistograms = -1;
    std::vector<cv::Mat> imagesHistograms(N_LABELS);
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

                // If not empty, compute histograms for the patch
                cv::Mat histograms;
                HistogramComparator::getImageHistograms(img, labelMask, histograms);

                if (numHistograms == -1)
                    numHistograms = histograms.cols;

                // Add histograms
                cv::Mat* curHistograms = &imagesHistograms[label];
                if (curHistograms->empty())
                    histograms.copyTo(*curHistograms);
                else
                    curHistograms->push_back(histograms);

                numProcessed++;
            }
        }

    std::cout << "Total processed patches: " << numProcessed << std::endl;
    std::cout << "Number of histograms: " << numHistograms << std::endl;

    // Compute average histograms for every label
    cv::Mat labelHistograms = cv::Mat(N_LABELS, numHistograms, CV_32F, cv::Scalar(0));
    for (int i = 0; i < labelHistograms.rows; i++)
        if (!imagesHistograms[i].empty()) {
            reduce(imagesHistograms[i], labelHistograms.row(i), 0, cv::REDUCE_AVG);
        }

    HistogramComparator::writeLabelHistogramsToFile(labelHistograms);
}