#include "../include/HistogramThresholder.h"

#include <opencv2/opencv.hpp>

const std::string HistogramThresholder::LABELS_HISTOGRAMS_PATH = "../data/labels_histograms.json";
const std::string HistogramThresholder::LABELS_HISTOGRAMS_NAME = "labelHistograms";

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

void HistogramThresholder::getImageHistogram(const cv::Mat& img, const cv::Mat& mask, cv::Mat& histogram) {
    getHueHistogram(img, mask, histogram);
    histogram.convertTo(histogram, DATA_TYPE);
}

void HistogramThresholder::writeLabelsHistogramsToFile(const cv::Mat& histograms) {
    cv::FileStorage fs(LABELS_HISTOGRAMS_PATH, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        std::cout << "Failed to open file " << LABELS_HISTOGRAMS_PATH << std::endl;
    fs << LABELS_HISTOGRAMS_NAME << histograms;
    fs.release();
}

void HistogramThresholder::readLabelsHistogramsFromFile(cv::Mat& histograms) {
    // Read template images
    cv::FileStorage fs(LABELS_HISTOGRAMS_PATH, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        std::cout << "Failed to open file " << LABELS_HISTOGRAMS_PATH << std::endl;
    fs[LABELS_HISTOGRAMS_NAME] >> histograms;
    fs.release();
}