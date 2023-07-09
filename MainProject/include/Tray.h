#pragma once
#include <opencv2/opencv.hpp>
#include "PlatesFinder.h"
#include "PlateRemover.h"
#include "FeatureComparator.h"

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
        
    public:
        const int MIN_FIRST_PLATE_LABEL = 1;
        const int MAX_FIRST_PLATE_LABEL = 5;

        const int SMALL_WINDOW_SIZE = 50;
        const int BIG_WINDOW_SIZE = 100;
        
        //
        Tray(std::string, std::string);
        
        //
        void FindPlates(const cv::Mat);
        
        //
        cv::Mat SegmentImage(const cv::Mat, std::vector<int>&, std::string);
        
        //
        cv::Mat DetectFoods(const cv::Mat);
        
        //
        cv::Mat SegmentFoods(const cv::Mat);

        //
        std::string get_traysAfterNames();

        //
        void PrintSaladPlate();
        
        void PrintInfo();
};