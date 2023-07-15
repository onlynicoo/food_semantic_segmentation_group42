#include "../include/FeatureComparator.h"

#include <opencv2/opencv.hpp>

const std::string FeatureComparator::LABEL_FEATURES_PATH = "../features/label_features.yml";
const std::string FeatureComparator::LABEL_FEATURES_NAME = "labelFeatures";

/**
 * The function `getLabelDistances` calculates the distances between a given image feature and a set of
 * label features, and returns the distances sorted in ascending order.
 * 
 * @param labelsFeatures A matrix containing the features of all labels. Each row represents the
 * features of a label.
 * @param labelWhitelist The `labelWhitelist` parameter is a vector of integers that specifies a list
 * of labels that should be included in the computation of label distances. Any label not present in
 * this whitelist will be excluded from the computation.
 * @param imgFeatures The `imgFeatures` parameter is a `cv::Mat` object that represents the features of
 * an image. It is used to calculate the distance between the image features and the features of each
 * label in `labelsFeatures`.
 * @return a vector of objects of type `FeatureComparator::LabelDistance`.
 */
std::vector<FeatureComparator::LabelDistance> FeatureComparator::getLabelDistances(
    const cv::Mat& labelsFeatures, std::vector<int> labelWhitelist, const cv::Mat& imgFeatures) {
    std::vector<FeatureComparator::LabelDistance> distances;
    for (int i = 0; i < labelsFeatures.rows; i++) {
        // Check if the label is excluded
        if (std::find(labelWhitelist.begin(), labelWhitelist.end(), i) == labelWhitelist.end())
            continue;

        double distance = cv::norm(labelsFeatures.row(i), imgFeatures, cv::NORM_L2);
        LabelDistance labelDistance;
        labelDistance.label = i;
        labelDistance.distance = distance;
        distances.push_back(labelDistance);
    }
    std::sort(distances.begin(), distances.end());
    return distances;
}

/**
 * The function "getHueFeatures" takes an input image and mask, converts it to HSV color space,
 * equalizes the value channel, computes the histogram of the hue channel using the specified number of
 * features, and normalizes the histogram.
 * 
 * @param img The input image in BGR format.
 * @param mask The mask parameter is a binary image that specifies which pixels in the input image
 * should be considered for computing the features. Only the pixels corresponding to non-zero values in
 * the mask will be used.
 * @param numFeatures The parameter "numFeatures" represents the number of histogram bins to be used
 * for computing the histogram. It determines the granularity of the histogram and affects the level of
 * detail in the resulting feature vector.
 * @param features The "features" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed histogram features.
 */
void FeatureComparator::getHueFeatures(const cv::Mat& img, const cv::Mat& mask, int numFeatures, cv::Mat& features) {
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
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numFeatures, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    features = hist.t();
}

/**
 * The function `getLBPFeatures` computes LBP texture features for an input image and mask, and stores
 * the result in the `features` matrix.
 * 
 * @param img The input image on which the LBP texture features will be computed. It is of type cv::Mat
 * and should be a grayscale image (CV_8UC1).
 * @param mask The "mask" parameter is a binary image of the same size as the input image "img". It is
 * used to specify which pixels in the input image should be considered for computing the LBP features.
 * Pixels with a value of 0 in the mask are skipped, while pixels with a non-zero
 * @param numFeatures The parameter "numFeatures" represents the number of bins or features in the
 * histogram. It determines the dimensionality of the resulting feature vector.
 * @param features The "features" parameter is an output parameter of type cv::Mat&. It is used to
 * store the computed LBP texture features.
 */
void FeatureComparator::getLBPFeatures(const cv::Mat& img, const cv::Mat& mask, int numFeatures, cv::Mat& features) {
    // Compute LBP texture features
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
    calcHist(&lbp_image, 1, 0, mask, hist, 1, &numFeatures, histRange);

    // Normalize the hist
    cv::normalize(hist, hist, NORMALIZE_VALUE, cv::NORM_L1);

    features = hist.t();
}

/**
 * The function getCannyLBPFeatures takes an input image, converts it to grayscale, applies Canny edge
 * detection, and then extracts LBP features from the resulting edges.
 * 
 * @param img The input image that you want to extract features from. It should be a 3-channel BGR
 * image.
 * @param mask The "mask" parameter is a binary image that specifies the region of interest in the
 * input image. Only the pixels corresponding to non-zero values in the mask will be considered for
 * feature extraction.
 * @param numFeatures The parameter "numFeatures" represents the number of LBP (Local Binary Patterns)
 * features to be extracted from the image.
 * @param features The "features" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed features obtained from the Canny edge detection and LBP (Local Binary Patterns) feature
 * extraction.
 */
void FeatureComparator::getCannyLBPFeatures(const cv::Mat& img, const cv::Mat& mask, int numFeatures, cv::Mat& features) {
    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    // Blur image
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);

    // Apply Canny edge detection
    cv::Mat edges;
    int t1 = 50, t2 = 150;
    Canny(grayImage, edges, t1, t2);

    getLBPFeatures(edges, mask, numFeatures, features);
}

/**
 * The function `getImageFeatures` calculates hue and canny LBP features for an image and combines them
 * into a single feature matrix.
 * 
 * @param img The input image on which the features will be extracted.
 * @param mask The "mask" parameter is a binary image of the same size as the input image "img". It is
 * used to specify which pixels in the input image should be considered for feature extraction. Pixels
 * with a value of 255 (white) in the mask are considered, while pixels with a value of
 * @param features The "features" parameter is an output parameter of type cv::Mat. It is used to store
 * the computed image features.
 */
void FeatureComparator::getImageFeatures(const cv::Mat& img, const cv::Mat& mask, cv::Mat& features) {
    cv::Mat hueFeatures, cannyLBPFeatures;
    getHueFeatures(img, mask, 64, hueFeatures);
    getCannyLBPFeatures(img, mask, 64, cannyLBPFeatures);

    features = 0.6 * hueFeatures;
    cv::hconcat(0.4 * cannyLBPFeatures, features, features);
}

/**
 * The function writes label features to a file using OpenCV's FileStorage.
 * 
 * @param features The "features" parameter is a cv::Mat object that represents the label features. It
 * is passed as a const reference to the writeLabelFeaturesToFile function.
 */
void FeatureComparator::writeLabelFeaturesToFile(const cv::Mat& features) {
    cv::FileStorage fs(LABEL_FEATURES_PATH, cv::FileStorage::WRITE);
    fs << LABEL_FEATURES_NAME << features;
    fs.release();
}

/**
 * The function reads label features from a file and stores them in a cv::Mat object.
 * 
 * @param features The "features" parameter is a reference to a cv::Mat object. It is used to store the
 * label features read from a file.
 */
void FeatureComparator::readLabelFeaturesFromFile(cv::Mat& features) {
    // Read template images
    cv::FileStorage fs(LABEL_FEATURES_PATH, cv::FileStorage::READ);
    if (!fs.isOpened())
        std::cout << "Failed to open label features file." << std::endl;

    fs[LABEL_FEATURES_NAME] >> features;
    fs.release();
}