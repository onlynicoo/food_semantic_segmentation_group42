#pragma once
#include <opencv2/opencv.hpp>

// Declaration of the class PlateFinder
class Tray {

    private:
        // Attributes

        // names of trays
        std::string traysBeforeNames;
        std::string traysAfterNames;

        // mats of trays
        cv::Mat traysBefore;
        cv::Mat traysAfter;

        std::string traysBeforeDetected;
        std::string traysAfterDetected;
        
        cv::Mat traysBeforeSegmented;
        cv::Mat traysAfterSegmented;
        
        void SaveSegmentedMask(std::string, cv::Mat);

        cv::Mat SegmentBread(cv::Mat);

    public:        
        //
        Tray(std::string, std::string);
        
        //
        void FindPlates(const cv::Mat);
        
        //
        cv::Mat SegmentImage(const cv::Mat&, std::vector<int>&, std::string);
        
        //
        cv::Mat SegmentFoods(const cv::Mat);

        //
        std::string get_trayAfterName();

        //
        void PrintSaladPlate();
        
        void ShowTray();
        void RefineSegmentation(const cv::Mat&, cv::Mat&, int);
};