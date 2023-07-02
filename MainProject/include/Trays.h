//#ifndef PLATES_FINDER // Header guard to prevent multiple inclusions
//#define PLATES_FINDER

#include <opencv2/opencv.hpp>

// Declaration of the class PlateFinder
class Trays {

  private:
    // Attributes

    // names of trays
    std::vector<std::string> traysBeforeNames;
    std::vector<std::string> traysAfterNames;

    // mats of trays
    std::vector<cv::Mat> traysBefore;
    std::vector<cv::Mat> traysAfter;

    std::vector<cv::Mat> traysBeforeDetected;
    std::vector<cv::Mat> traysAfterDetected;
    
    std::vector<cv::Mat> traysBeforeSegmented;
    std::vector<cv::Mat> traysAfterSegmented;
    
    // trays number
    int traysNumber;

  public:
    Trays();
    Trays(std::string, std::string);
    void AddTray(std::string, std::string);
    void FindPlates(const cv::Mat);

    cv::Mat DetectFoods(const cv::Mat);
    cv::Mat SegmentFoods(const cv::Mat);
    void PrintInfo();
};

//#endif // End of header guard