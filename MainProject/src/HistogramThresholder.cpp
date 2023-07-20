#include "../include/HistogramThresholder.h"

#include <fstream>
#include <opencv2/opencv.hpp>

const std::string HistogramThresholder::LABELS_HISTOGRAMS_PATH = "../data/labels_histograms.txt";

/**
 * The function "getLabelDistances" calculates the distances between the histograms of labels and an
 * image, and returns a sorted vector of label-distance pairs.
 * 
 * @param labelsHistograms A matrix containing histograms of labels. Each row represents a label and
 * each column represents a bin in the histogram.
 * @param labelWhitelist A vector of integers representing the labels that should be included in the
 * calculation of distances. Labels not present in this whitelist will be excluded from the
 * calculation.
 * @param imgHistogram The `imgHistogram` parameter is a `cv::Mat` object representing the histogram of
 * an image. It is used to compare the histograms of different labels in the `labelsHistograms` matrix.
 * @return a vector of HistogramThresholder::LabelDistance objects.
 */
std::vector<HistogramThresholder::LabelDistance> HistogramThresholder::getLabelDistances(
    const cv::Mat& labelsHistograms, std::vector<int> labelWhitelist, const cv::Mat& imgHistogram) {
    
    std::vector<HistogramThresholder::LabelDistance> distances;
    for (int i = 0; i < labelsHistograms.rows; i++) {
        
        // Check if the label is excluded
        if (std::find(labelWhitelist.begin(), labelWhitelist.end(), i) == labelWhitelist.end())
            continue;

        double distance = cv::norm(labelsHistograms.row(i), imgHistogram, cv::NORM_L2);
        LabelDistance labelDistance;
        labelDistance.label = i;
        labelDistance.distance = distance;
        distances.push_back(labelDistance);
    }
    std::sort(distances.begin(), distances.end());
    return distances;
}

/**
 * The function `getHueHistogram` takes an input image and mask, converts it to HSV color space,
 * equalizes the value channel, computes the histogram of the hue channel, and normalizes the
 * histogram.
 * 
 * @param img The input image in BGR format.
 * @param mask The mask parameter is a binary image that specifies which pixels in the input image
 * should be included in the histogram calculation. Only the pixels corresponding to non-zero values in
 * the mask will be considered.
 * @param histograms The `histograms` parameter is an output parameter of type `cv::Mat`. It is used to
 * store the computed histogram of the hue channel.
 */
void HistogramThresholder::getHueHistogram(const cv::Mat& img, const cv::Mat& mask, cv::Mat& histograms) {
    
    // Convert to HSV
    cv::Mat hsvImg;
    std::vector<cv::Mat> hsvChannels;
    cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
    split(hsvImg, hsvChannels);

    // Equalize V channel to enhance color difference
    cv::Mat valueChannel;
    hsvChannels[2].copyTo(valueChannel, mask);
    cv::equalizeHist(valueChannel, valueChannel);
    valueChannel.copyTo(hsvChannels[2], mask);

    // Merge back the channels and convert back to BGR
    cv::Mat modHsvImg, modBgrImg;
    cv::merge(hsvChannels, modHsvImg);

    // Convert to HSV
    cv::Mat hueChannel;
    cvtColor(modHsvImg, hsvImg, cv::COLOR_BGR2HSV);
    cv::split(hsvImg, hsvChannels);
    hueChannel = hsvChannels[0];

    // Compute hist
    float range[] = {0, 180};
    const float* histRange[] = {range};
    int numBins = HistogramThresholder::NUM_VALUES;
    cv::Mat hist;
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numBins, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    histograms = hist.t();
}

/**
 * The function `getImageHistogram` calculates the histogram of an image using a given mask and
 * converts it to a specified data type.
 * 
 * @param img The input image for which the histogram needs to be computed.
 * @param mask The "mask" parameter is a binary image of the same size as the input image "img". It is
 * used to specify which pixels of the input image should be considered for computing the histogram.
 * Pixels with a non-zero value in the mask are included in the histogram calculation, while pixels
 * with a zero
 * @param histogram The `histogram` parameter is an output parameter of type `cv::Mat`. It is used to
 * store the computed histogram of the image.
 */
void HistogramThresholder::getImageHistogram(const cv::Mat& img, const cv::Mat& mask, cv::Mat& histogram) {
    getHueHistogram(img, mask, histogram);
    histogram.convertTo(histogram, DATA_TYPE);
}

/**
 * The function writes the labels histograms to a file.
 * 
 * @param histograms The parameter `histograms` is a `cv::Mat` object, which represents a matrix
 * containing the histograms. It is passed by reference to the `writeLabelsHistogramsToFile` function.
 */
void HistogramThresholder::writeLabelsHistogramsToFile(const cv::Mat& histograms) {
    std::ofstream out(LABELS_HISTOGRAMS_PATH);
    if (!out.is_open())
        std::cout << "Error creating " << LABELS_HISTOGRAMS_PATH << std::endl;
        
    out << histograms.rows << std::endl << histograms.cols << std::endl;
    for (int r = 0; r < histograms.rows; r++) {
        for (int c = 0; c < histograms.cols; c++)
            out << int(histograms.at<uchar>(r,c)) << " ";
        out << std::endl;
    }
    out.close();
}

/**
 * The function reads histograms from a file and stores them in a cv::Mat object.
 * 
 * @param histograms A reference to a cv::Mat object that will store the histograms read from the file.
 */
void HistogramThresholder::readLabelsHistogramsFromFile(cv::Mat& histograms) {
    std::ifstream in(LABELS_HISTOGRAMS_PATH);
    if (!in.is_open())
        std::cout << "Error opening " << LABELS_HISTOGRAMS_PATH << std::endl;

    int curValue, curRow = -1, rows = -1, cols = -1;
    std::string line;
    while (std::getline(in, line)) {
        int curCol = 0;
        std::istringstream ss(line);
        while (ss >> curValue) {
            if (rows == -1) {
                rows = curValue;
            } else {
                if (cols == -1) {
                    cols = curValue;
                    histograms = cv::Mat(rows, cols, DATA_TYPE);
                } else {
                    histograms.at<uchar>(curRow, curCol) = curValue;
                    curCol++;
                }
            }
        }
        if (rows != -1 && cols != -1)
            curRow++;
    }
    in.close();
}