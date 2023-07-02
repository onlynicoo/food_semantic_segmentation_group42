#include <iostream>
#include <opencv2/opencv.hpp>
#include <FeatureComparator.h>

using namespace cv;
using namespace std;

int FeatureComparator::getFoodLabel(Mat labelsFeatures, vector<int> excludedLabels, Mat imgFeatures) {
    double minDistance = DBL_MAX;
    int nearestLabelIdx = -1;
    for (int i = 0; i < labelsFeatures.rows; i++) {
        
        // Check if the label is excluded
        if (find(excludedLabels.begin(), excludedLabels.end(), i) != excludedLabels.end())
            continue;

        Mat curFeatures = labelsFeatures.row(i);
        double distance = norm(curFeatures, imgFeatures, NORM_L2);
        if (distance < minDistance) {
            minDistance = distance;
            nearestLabelIdx = i;
        }
    }
    return nearestLabelIdx;
}

Mat FeatureComparator::getHueFeatures(Mat img, Mat mask, int numFeatures) {
    // Convert to HSV
    Mat hsvImg;
    vector<Mat> hsvChannels;
    cvtColor(img, hsvImg, COLOR_BGR2HSV); 
    split(hsvImg, hsvChannels);

    // Equalize V channel to enhance color difference
    Mat valueChannel;
    hsvChannels[2].copyTo(valueChannel, mask);
    equalizeHist(valueChannel, valueChannel);
    valueChannel.copyTo(hsvChannels[2], mask);

    // Merge back the channels and convert back to BGR
    Mat modHsvImg, modBgrImg;
    merge(hsvChannels, modHsvImg);

    // Convert to HSV
    Mat hueChannel;
    cvtColor(modHsvImg, hsvImg, COLOR_BGR2HSV);
    split(hsvImg, hsvChannels);
    hueChannel = hsvChannels[0];
    
    // Compute hist
    float range[] = {0, 180};
    const float* histRange[] = {range};
    Mat hist;
    calcHist(&hueChannel, 1, 0, mask, hist, 1, &numFeatures, histRange);
    
    // Normalize the hist
    hist /= sum(hist)[0];

    return hist.t();
}

Mat FeatureComparator::getLBPFeatures(Mat img, Mat mask, int numFeatures) {
    // Compute LBP texture features
    int lbp_radius = 1;
    int lbp_neighbors = pow(lbp_radius * 2 + 1, 2) - 1;
    Mat lbp_image = Mat::zeros(img.size(), CV_8UC1);

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
    Mat hist;
    calcHist(&lbp_image, 1, 0, mask, hist, 1, &numFeatures, histRange);

    // Normalize the hist
    hist /= sum(hist)[0];
    
    return hist.t();
}

Mat FeatureComparator::getCannyLBPFeatures(Mat img, Mat mask, int numFeatures) {

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);

    // Blur image
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);

    // Apply Canny edge detection
    Mat edges;
    int t1 = 50, t2 = 150;
    Canny(grayImage, edges, t1, t2);

    return getLBPFeatures(edges, mask, numFeatures);
}

void FeatureComparator::appendColumns(Mat src, Mat &dst) {
    if (dst.empty())
        dst = src;
    else
        hconcat(src, dst, dst);
}

Mat FeatureComparator::getImageFeatures(Mat img, Mat mask) {
    Mat features;
    appendColumns(0.6 * getHueFeatures(img, mask, 64), features);
    appendColumns(0.4 * getCannyLBPFeatures(img, mask, 64), features);
    return features;
}

/*
Mat getTextureFeatures(Mat img, Mat mask, int numFeatures) {
    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);

    // Create a HOG descriptor object
    cv::HOGDescriptor hog;

    // Set the HOG parameters
    cv::Size winSize(64, 128);
    cv::Size blockSize(32, 32);
    cv::Size blockStride(8, 8);
    cv::Size cellSize(8, 8);
    int numBins = 9;

    // Set the parameters in the HOG descriptor object
    hog.winSize = winSize;
    hog.blockSize = blockSize;
    hog.blockStride = blockStride;
    hog.cellSize = cellSize;
    hog.nbins = numBins;

    // Extract the regions indicated by the mask
    cv::Mat maskedImage;
    grayImage.copyTo(maskedImage, mask);

    // Compute the HOG descriptor for the masked image
    std::vector<float> descriptors;
    hog.compute(maskedImage, descriptors);

    // Convert the HOG descriptors to a cv::Mat object
    cv::Mat hogMat(descriptors);

    // Reshape the cv::Mat to a single row
    hogMat = hogMat.reshape(1, 1);
    
    return hogMat;
}

Mat getSIFTFeatures(Mat img, Mat mask) {
    // Detect and compute
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(img, mask, keypoints, descriptors);

    cout << "Detected keypoints: " << keypoints.size() << endl;
    drawKeypoints(cropInputImg, keypoints, outImg);
    imshow("out", outImg);
    waitKey();

    // Compute mean descriptor vector for the image
    Mat res(1, descriptors.cols, CV_32F, Scalar(0));
    for (int c = 0; c < descriptors.cols; c++) {
        float colSum = 0;
        for (int r = 0; r < descriptors.rows; r++)
            colSum += descriptors.at<float>(r,c);
        res.at<float>(0,c) = colSum / float(descriptors.rows);
    }

    return res;
}

Mat getHistImage(Mat hist) {
    int hist_size = hist.rows, hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / hist_size);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hist_size; i++) {
        line(
            histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            Scalar(255, 255, 255),
            2, 8, 0);
    }
    return histImage;
}
*/