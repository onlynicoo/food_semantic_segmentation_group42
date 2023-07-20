// Author: Nicola Lorenzon, Daniele Moschetta

#pragma once
#include <opencv2/opencv.hpp>

// Handles the functionalities of tray before and after images
class Tray {

    private:
        // Attributes

        // Trays names
        std::string trayBeforePath;
        std::string trayAfterPath;

        // Trays images
        cv::Mat trayBeforeImage;
        cv::Mat trayAfterImage;

        // Trays bounding boxes
        std::string trayBeforeBoundingBoxesPath;
        std::string trayAfterBoundingBoxesPath;
        
        // Trays images
        cv::Mat trayBeforeSegmentationMask;
        cv::Mat trayAfterSegmentationMask;
        
        // Function to save the just computed segmentation mask
        void saveSegmentedMask(const std::string&, const cv::Mat&);

    public:

        // Constructor that orchestrate the flow of segmentation and detection.
        Tray(const std::string&, const std::string&);
        
        // Segment a given image, saving the food found and the the bounding boxes
        void segmentImage(const cv::Mat&, cv::Mat&, std::vector<int>&, const std::string&);
        
        // Get trayAfterPath
        std::string getTrayAfterPath();
        
        // Show the trays before and after the meal, the detection bounding boxes and the segmentation masks
        void showTray();

        // Function to print out food quantity
        void printFoodQuantities();

};