#include <iostream>
#include <opencv2/opencv.hpp>

class FeatureComparator {
    private:
        cv::Mat templateFeatures;
        
        cv::Mat getCannyLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        cv::Mat getHueFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        cv::Mat getLBPFeatures(cv::Mat img, cv::Mat mask, int numFeatures);
        int findNearestCenter(cv::Mat centers, cv::Mat dataRow);
        int getLabelFromImgPath(std::string path);
        cv::Mat getImageFeatures(cv::Mat img, cv::Mat mask);
        cv::Mat getTemplateFeatures(std::string templateDir);
        void appendColumns(cv::Mat src, cv::Mat &dst);
        void preProcessTemplateImage(cv::Mat &img, cv::Mat &mask);

    public:
        FeatureComparator();
        FeatureComparator(std::string templateDir);
        int getFoodLabel(cv::Mat img, cv::Mat mask);
};