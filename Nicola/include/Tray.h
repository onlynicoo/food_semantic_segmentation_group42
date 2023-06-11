//#ifndef TRAY_H // Header guard to prevent multiple inclusions
//#define TRAY_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct my_line
{
  cv::Point p1;
  cv::Point p2;

  my_line(cv::Point a, cv::Point b) {
    p1 = a;
    p2 = b;
  }
};


// Declaration of the Example class
class Tray {

  private:
    // Attributes
    // line vectors: right | left | top | bottom

    std::vector<std::vector<Point>> true_lines;    
    
    std::vector<std::vector<Point>> wider_lines;

  public:
    // Generic constructor
    Tray(int r, int c);
    
    // Add a line
    void addLine(bool vertical, Point p1, Point p2);

    // Getter
    std::vector<std::vector<Point>> getTray();

    void update_vertical_lines(Point pt1, Point pt2);
    void update_horizontal_lines(Point pt1, Point pt2);

    // toString
    std::string toString();
};

//#endif // End of header guard