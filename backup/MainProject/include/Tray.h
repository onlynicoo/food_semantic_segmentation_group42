//#ifndef PLATES_FINDER // Header guard to prevent multiple inclusions
//#define PLATES_FINDER

#include <opencv2/opencv.hpp>
#include "PlatesFinder.h"

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

    cv::Mat traysBeforeDetected;
    cv::Mat traysAfterDetected;
    
    cv::Mat traysBeforeSegmented;
    cv::Mat traysAfterSegmented;
    
    // trays number
    int traysNumber;

  public:
    //Tray();
    Tray(std::string, std::string);
    //void AddTray(std::string, std::string);
    void FindPlates(const cv::Mat);

    void ElaborateImage(const cv::Mat, cv::Mat[]);
    cv::Mat DetectFoods(const cv::Mat);
    cv::Mat SegmentFoods(const cv::Mat);

    void PrintInfo();
};

//#endif // End of header guard