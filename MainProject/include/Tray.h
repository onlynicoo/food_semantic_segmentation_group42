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

        std::string LABELS[14] = {
        "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
        "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
        "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

        std::string labelFeaturesPath = "../features/label_features.yml";
        
        void SaveSegmentedMask(std::string, cv::Mat);

        cv::Mat SegmentBread(cv::Mat);

    public:
        const int SMALL_WINDOW_SIZE = 50;
        const int BIG_WINDOW_SIZE = 150;
        
        //
        Tray(std::string, std::string);
        
        //
        void FindPlates(const cv::Mat);
        
        //
        cv::Mat SegmentImage(const cv::Mat, std::vector<int>&, std::string);
        
        //
        cv::Mat SegmentFoods(const cv::Mat);

        //
        std::string get_traysAfterNames();

        //
        void PrintSaladPlate();
        
        void PrintInfo();
        void RefineSegmentation(const cv::Mat&, cv::Mat&, int);
        //void RefineSegmentation();

};