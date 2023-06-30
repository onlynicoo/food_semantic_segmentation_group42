#include <iostream>
#include <opencv2/opencv.hpp>
#include <FeatureComparator.h>

using namespace cv;
using namespace std;

int N_LABELS = 11;
Size STD_SIZE(256, 256);

FeatureComparator::FeatureComparator() {
    templateFeatures = Mat();
}

FeatureComparator::FeatureComparator(string templateDir) {
    templateFeatures = getTemplateFeatures(templateDir);
}

int FeatureComparator::getFoodLabel(Mat img, Mat mask) {
    Mat imgFeatures = getImageFeatures(img, mask);
    int label = findNearestCenter(templateFeatures, imgFeatures) + 1;
    return label;
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

void FeatureComparator::preProcessTemplateImage(Mat &img, Mat &mask) {
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

    resize(img, img, STD_SIZE);
    resize(mask, mask, STD_SIZE);
}

int FeatureComparator::getLabelFromImgPath(string path) {
    string name = path.substr(path.find_last_of('\\') + 1, path.find_last_of(".") - 1 - path.find_last_of('\\'));
    int label = stoi(name.substr(2, name.find_first_of('_') - 2));
    return label;
}

Mat FeatureComparator::getTemplateFeatures(string templateDir) {

    // Read file names
    vector<String> inputNames;
    glob(templateDir + "/*.png", inputNames, false);
    cout << "Input images found: " << inputNames.size() << endl;

    // Skip 12 (salad) and 13 (bread) since they should have been already distinguished
    for (int i = 0; i < inputNames.size(); i++)
        if (getLabelFromImgPath(inputNames[i]) > N_LABELS) {
            inputNames.erase(inputNames.begin() + i);
            i--;
        }

    Mat img, mask;
    int numProcessed = 0;
    Mat allFeatures;
    vector<Mat> imagesFeatures(N_LABELS);
    for (int i = 0; i < inputNames.size(); i++) {

        if (numProcessed != 0 && numProcessed % 100 == 0)
            cout << numProcessed << " images processed " << endl;
        
        //cout << "Processing image: " << inputNames[i] << endl;

        img = imread(inputNames[i], IMREAD_UNCHANGED);
        preProcessTemplateImage(img, mask);

        Mat features = getImageFeatures(img, mask);
        Mat* curFeatures = &imagesFeatures[getLabelFromImgPath(inputNames[i]) - 1];
        if (curFeatures->empty())
            features.copyTo(*curFeatures);
        else
            curFeatures->push_back(features);

        numProcessed++;
    }

    cout << "Total images processed: " << numProcessed << endl;
    cout << "Number of features: " << imagesFeatures[0].cols << endl;

    // Compute average features for every class
    Mat classesFeatures = Mat(N_LABELS, imagesFeatures[0].cols, CV_32F, Scalar(0));
    for (int i = 0; i < classesFeatures.rows; i++)
        if (!imagesFeatures[i].empty()) {
            reduce(imagesFeatures[i], classesFeatures.row(i), 0, REDUCE_AVG);
        }
    return classesFeatures;
}

int FeatureComparator::findNearestCenter(Mat centers, Mat features) {
    double minDistance = DBL_MAX;
    int nearestCenterIndex = -1;
    for (int i = 0; i < centers.rows; i++) {
        Mat center = centers.row(i);
        double distance = norm(center, features, cv::NORM_L2);
        // cout << i + 1 << " " << distance << endl;
        if (distance < minDistance) {
            minDistance = distance;
            nearestCenterIndex = i;
        }
    }
    return nearestCenterIndex;
}

/*
    ### NOT USED ###
*/

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

    /*cout << "Detected keypoints: " << keypoints.size() << endl;
    drawKeypoints(cropInputImg, keypoints, outImg);
    imshow("out", outImg);
    waitKey();*/

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
