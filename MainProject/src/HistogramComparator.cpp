#include "../include/HistogramComparator.h"

#include <opencv2/opencv.hpp>

const std::string HistogramComparator::LABEL_HISTOGRAMS_PATH = "../data/label_histograms.json";
const std::string HistogramComparator::LABEL_HISTOGRAMS_NAME = "labelHistograms";

/**
 * The function `getLabelDistances` calculates the distances between a given image histogram and a set of
 * label histograms, and returns the distances sorted in ascending order.
 * 
 * @param labelsHistograms A matrix containing the histograms of all labels. Each row represents the
 * histograms of a label.
 * @param labelWhitelist The `labelWhitelist` parameter is a vector of integers that specifies a list
 * of labels that should be included in the computation of label distances. Any label not present in
 * this whitelist will be excluded from the computation.
 * @param imgHistograms The `imgHistograms` parameter is a `cv::Mat` object that represents the histograms of
 * an image. It is used to calculate the distance between the image histograms and the histograms of each
 * label in `labelsHistograms`.
 * @return a vector of objects of type `HistogramComparator::LabelDistance`.
 */
std::vector<HistogramComparator::LabelDistance> HistogramComparator::getLabelDistances(
    const cv::Mat& labelsHistograms, std::vector<int> labelWhitelist, const cv::Mat& imgHistograms) {
    std::vector<HistogramComparator::LabelDistance> distances;
    for (int i = 0; i < labelsHistograms.rows; i++) {
        // Check if the label is excluded
        if (std::find(labelWhitelist.begin(), labelWhitelist.end(), i) == labelWhitelist.end())
            continue;

        double distance = cv::norm(labelsHistograms.row(i), imgHistograms, cv::NORM_L2);
        LabelDistance labelDistance;
        labelDistance.label = i;
        labelDistance.distance = distance;
        distances.push_back(labelDistance);
    }
    std::sort(distances.begin(), distances.end());
    return distances;
}

/**
 * The function "getHueHistograms" takes an input image and mask, converts it to HSV color space,
 * equalizes the value channel, computes the histogram of the hue channel using the specified number of
 * histograms, and normalizes the histogram.
 * 
 * @param img The input image in BGR format.
 * @param mask The mask parameter is a binary image that specifies which pixels in the input image
 * should be considered for computing the histograms. Only the pixels corresponding to non-zero values in
 * the mask will be used.
 * @param numHistograms The parameter "numHistograms" represents the number of histogram bins to be used
 * for computing the histogram. It determines the granularity of the histogram and affects the level of
 * detail in the resulting histogram vector.
 * @param histograms The "histograms" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed histogram histograms.
 */
void HistogramComparator::getHueHistograms(const cv::Mat& img, const cv::Mat& mask, int numHistograms, cv::Mat& histograms) {
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
    cv::Mat hist;
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numHistograms, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    histograms = hist.t();
}

/**
 * The function `getLBPHistograms` computes LBP texture histograms for an input image and mask, and stores
 * the result in the `histograms` matrix.
 * 
 * @param img The input image on which the LBP texture histograms will be computed. It is of type cv::Mat
 * and should be a grayscale image (CV_8UC1).
 * @param mask The "mask" parameter is a binary image of the same size as the input image "img". It is
 * used to specify which pixels in the input image should be considered for computing the LBP histograms.
 * Pixels with a value of 0 in the mask are skipped, while pixels with a non-zero
 * @param numHistograms The parameter "numHistograms" represents the number of bins or histograms in the
 * histogram. It determines the dimensionality of the resulting histogram vector.
 * @param histograms The "histograms" parameter is an output parameter of type cv::Mat&. It is used to
 * store the computed LBP texture histograms.
 */
void HistogramComparator::getLBPHistograms(const cv::Mat& img, const cv::Mat& mask, int numHistograms, cv::Mat& histograms) {
    // Compute LBP texture histograms
    int lbp_radius = 1;
    int lbp_neighbors = pow(lbp_radius * 2 + 1, 2) - 1;
    cv::Mat lbp_image = cv::Mat::zeros(img.size(), CV_8UC1);

    for (int y = lbp_radius; y < img.rows - lbp_radius; y++) {
        for (int x = lbp_radius; x < img.cols - lbp_radius; x++) {
            // Skip not maked pixels
            if (mask.at<uchar>(y, x) == 0)
                continue;

            uchar center_value = img.at<uchar>(y, x);
            uchar lbp_code = 0;
            for (int i = 0; i < lbp_neighbors; i++) {
                float angle = 2 * CV_PI * i / lbp_neighbors;
                int x_i = int(x + lbp_radius * cos(angle));
                int y_i = int(y + lbp_radius * sin(angle));

                uchar neighbor_value = img.at<uchar>(y_i, x_i);
                if (neighbor_value >= center_value)
                    lbp_code++;
            }

            lbp_image.at<uchar>(y, x) = lbp_code;
        }
    }

    // Compute hist
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::Mat hist;
    calcHist(&lbp_image, 1, 0, mask, hist, 1, &numHistograms, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    histograms = hist.t();
}

/**
 * The function getCannyLBPHistograms takes an input image, converts it to grayscale, applies Canny edge
 * detection, and then extracts LBP histograms from the resulting edges.
 * 
 * @param img The input image that you want to extract histograms from. It should be a 3-channel BGR
 * image.
 * @param mask The "mask" parameter is a binary image that specifies the region of interest in the
 * input image. Only the pixels corresponding to non-zero values in the mask will be considered for
 * histogram extraction.
 * @param numHistograms The parameter "numHistograms" represents the number of LBP (Local Binary Patterns)
 * histograms to be extracted from the image.
 * @param histograms The "histograms" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed histograms obtained from the Canny edge detection and LBP (Local Binary Patterns) histogram
 * extraction.
 */
void HistogramComparator::getCannyLBPHistograms(const cv::Mat& img, const cv::Mat& mask, int numHistograms, cv::Mat& histograms) {
    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    // Blur image
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);

    // Apply Canny edge detection
    cv::Mat edges;
    int t1 = 50, t2 = 150;
    Canny(grayImage, edges, t1, t2);

    getLBPHistograms(edges, mask, numHistograms, histograms);
}

/**
 * The function `getImageHistograms` calculates hue and canny LBP histograms for an image and combines them
 * into a single histogram matrix.
 * 
 * @param img The input image on which the histograms will be extracted.
 * @param mask The "mask" parameter is a binary image of the same size as the input image "img". It is
 * used to specify which pixels in the input image should be considered for histogram extraction. Pixels
 * with a value of 255 (white) in the mask are considered, while pixels with a value of
 * @param histograms The "histograms" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed image histograms.
 */
void HistogramComparator::getImageHistograms(const cv::Mat& img, const cv::Mat& mask, cv::Mat& histograms) {
    cv::Mat hueHistograms, cannyLBPHistograms;
    getHueHistograms(img, mask, 64, hueHistograms);
    getCannyLBPHistograms(img, mask, 64, cannyLBPHistograms);

    histograms = 0.6 * hueHistograms;
    cv::hconcat(0.4 * cannyLBPHistograms, histograms, histograms);
}

/**
 * The function writes label histograms to a file using OpenCV's FileStorage.
 * 
 * @param histograms The "histograms" parameter is a cv::Mat object that represents the label histograms. It
 * is passed as a const reference to the writeLabelHistogramsToFile function.
 */
void HistogramComparator::writeLabelHistogramsToFile(const cv::Mat& histograms) {
    cv::FileStorage fs(LABEL_HISTOGRAMS_PATH, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs << LABEL_HISTOGRAMS_NAME << histograms;
    fs.release();
}

/**
 * The function reads label histograms from a file and stores them in a cv::Mat object.
 * 
 * @param histograms The "histograms" parameter is a reference to a cv::Mat object. It is used to store the
 * label histograms read from a file.
 */
void HistogramComparator::readLabelHistogramsFromFile(cv::Mat& histograms) {
    // Read template images
    cv::FileStorage fs(LABEL_HISTOGRAMS_PATH, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        std::cout << "Failed to open label histograms file." << std::endl;

    fs[LABEL_HISTOGRAMS_NAME] >> histograms;
    fs.release();
}