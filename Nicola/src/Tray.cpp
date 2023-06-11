#include "../include/Tray.h" // Include the corresponding header file

//using namespace std;
using namespace cv;

// Implementation of the Example class constructor
//Tray::Tray(int r, int c) {
    /*
    // Initialize points at the center of img
    left = Point(int(c/2), int(r/2));
    rv = Point(int(c/2), int(r/2));
    th = Point(int(c/2), int(r/2));
    bh = Point(int(c/2), int(r/2));
    */

//    std::cout << "a";
//}

//Tray::~Tray() {
Tray::Tray(int r, int c) {
    /*
    Point w_c1 = Point(0,0);
    Point w_c2 = Point(r,0);
    Point w_c3 = Point(0,c);
    Point w_c4 = Point(r,c);
    */

    Point t_c1 = Point(int(r/2)-1,int(c/2)-1);
    Point t_c2 = Point(int(r/2)+1,int(c/2)-1);
    Point t_c3 = Point(int(r/2)-1,int(c/2)+1);
    Point t_c4 = Point(int(r/2)+1,int(c/2)-1);
    /*
    wider_lines.push_back(std::vector<Point>{w_c1, w_c2});
    wider_lines.push_back(std::vector<Point>{w_c2, w_c4});
    wider_lines.push_back(std::vector<Point>{w_c1, w_c3});
    wider_lines.push_back(std::vector<Point>{w_c3, w_c4});
    */

    wider_lines.push_back(std::vector<Point>{t_c1, t_c2});
    wider_lines.push_back(std::vector<Point>{t_c2, t_c4});
    wider_lines.push_back(std::vector<Point>{t_c1, t_c3});
    wider_lines.push_back(std::vector<Point>{t_c3, t_c4});

    true_lines.push_back(std::vector<Point>{t_c1, t_c2});
    true_lines.push_back(std::vector<Point>{t_c2, t_c4});
    true_lines.push_back(std::vector<Point>{t_c1, t_c3});
    true_lines.push_back(std::vector<Point>{t_c3, t_c4});
}

// Implementation of a member function of the Tray class
std::string Tray::toString() {
    // Function code
    bool format = 0;
    std::string str;
    if (!format) {
        str = 
            "lv p1 = (" + std::to_string(true_lines[0][0].x) + ", " + std::to_string(true_lines[0][0].y) +") \n" +
            "lv p2 = (" + std::to_string(true_lines[0][1].x) + ", " + std::to_string(true_lines[0][1].y) +") \n" + 
            "rv p1 = (" + std::to_string(true_lines[1][0].x) + ", " + std::to_string(true_lines[1][0].y) +") \n" +
            "rv p2 = (" + std::to_string(true_lines[1][1].x) + ", " + std::to_string(true_lines[1][1].y) +") \n" +
            "th p1 = (" + std::to_string(true_lines[2][0].x) + ", " + std::to_string(true_lines[2][0].y) +") \n" +
            "th p2 = (" + std::to_string(true_lines[2][1].x) + ", " + std::to_string(true_lines[2][1].y) +") \n" +
            "bh p1 = (" + std::to_string(true_lines[3][0].x) + ", " + std::to_string(true_lines[3][0].y) +") \n" +
            "bh p2 = (" + std::to_string(true_lines[3][1].x) + ", " + std::to_string(true_lines[3][1].y) +") \n";
    }
    else {
        str =
            "lv p1.x = " + std::to_string(true_lines[0][0].x) + "\n" + 
            "lv p1.y = " + std::to_string(true_lines[0][0].y) + "\n" + 
            "lv p2.x = " + std::to_string(true_lines[0][1].x) + "\n" +
            "lv p2.y = " + std::to_string(true_lines[0][1].y) + "\n" +
            
            "rv p1.x = " + std::to_string(true_lines[1][0].x) + "\n" +
            "rv p1.y = " + std::to_string(true_lines[1][0].y) + "\n" +
            "rv p2.x = " + std::to_string(true_lines[1][1].x) + "\n" +
            "rv p2.y = " + std::to_string(true_lines[1][1].y) + "\n" +
            
            "th p1.x = " + std::to_string(true_lines[2][0].x) + "\n" +
            "th p1.y = " + std::to_string(true_lines[2][0].y) + "\n" +
            "th p2.x = " + std::to_string(true_lines[2][1].x) + "\n" +
            "th p2.y = " + std::to_string(true_lines[2][1].y) + "\n" +
            
            "bh p1.x = " + std::to_string(true_lines[3][0].x) + "\n" +
            "bh p1.y = " + std::to_string(true_lines[3][0].y) + "\n" +
            "bh p2.x = " + std::to_string(true_lines[3][1].x) + "\n" +
            "bh p2.y = " + std::to_string(true_lines[3][1].y) + "\n";
    }
   return str;
}


// Implementation of a member function of the Tray class
std::vector<std::vector<Point>> Tray::getTray() {
    try {
        return true_lines;
    }
    catch(Exception e) {
        return true_lines;
    }
}

/*
bool isLineOnLeft(const cv::Point& line1Start, const cv::Point& line1End,
                  const cv::Point& line2Start, const cv::Point& line2End) {
    // Calculate the cross product of the two lines
    int crossProduct = (line2End.x - line2Start.x) * (line1End.y - line1Start.y) -
                       (line2End.y - line2Start.y) * (line1End.x - line1Start.x);

    // Check if the cross product is positive, indicating that line1 is on the left side of line2
    return crossProduct > 0;
}
*/
    
void Tray::update_vertical_lines(Point pt1, Point pt2) {         
    if (pt1.y < true_lines[0][0].y || pt1.y < true_lines[0][1].y) {
        if (pt1.y < wider_lines[0][0].y || pt1.y < wider_lines[0][1].y) {
            true_lines[0][0] = wider_lines[0][0];
            true_lines[0][1] = wider_lines[0][1];

            wider_lines[0][0].x = pt1.x;
            wider_lines[0][0].y = pt1.y;
            wider_lines[0][1].x = pt2.x;
            wider_lines[0][1].y = pt2.y;
        }
        else {
            true_lines[0][0].x = pt1.x;
            true_lines[0][0].y = pt1.y;
            true_lines[0][1].x = pt2.x;
            true_lines[0][1].y = pt2.y;
        }
    }
    if (pt1.y > true_lines[1][0].y || pt1.y > true_lines[1][1].y) {
        if (pt1.y > wider_lines[1][0].y || pt1.y > wider_lines[1][1].y) {
            true_lines[1][0] = wider_lines[1][0];
            true_lines[1][1] = wider_lines[1][1];

            wider_lines[1][0].x = pt1.x;
            wider_lines[1][0].y = pt1.y;
            wider_lines[1][1].x = pt2.x;
            wider_lines[1][1].y = pt2.y;
        }
        else {
            true_lines[1][0].x = pt1.x;
            true_lines[1][0].y = pt1.y;
            true_lines[1][1].x = pt2.x;
            true_lines[1][1].y = pt2.y;
        }
    }
}

void Tray::update_horizontal_lines(Point pt1, Point pt2) {
    // check if this line has x or y coord 
    if (pt1.x < true_lines[2][0].x || pt1.x < true_lines[2][1].x) {
        if (pt1.x < wider_lines[2][0].x || pt1.x < wider_lines[2][1].x) {
            true_lines[2][0] = wider_lines[2][0];
            true_lines[2][1] = wider_lines[2][1];

            wider_lines[2][0].x = pt1.x;
            wider_lines[2][0].y = pt1.y;
            wider_lines[2][1].x = pt2.x;
            wider_lines[2][1].y = pt2.y;
        }
        else {
            true_lines[2][0].x = pt1.x;
            true_lines[2][0].y = pt1.y;
            true_lines[2][1].x = pt2.x;
            true_lines[2][1].y = pt2.y;
        }
    }
    if (pt1.x > true_lines[3][0].x || pt1.x > true_lines[3][1].x) {
        if (pt1.x > wider_lines[3][0].x || pt1.x > wider_lines[3][1].x) {
            true_lines[3][0] = wider_lines[3][0];
            true_lines[3][1] = wider_lines[3][1];

            wider_lines[3][0].x = pt1.x;
            wider_lines[3][0].y = pt1.y;
            wider_lines[3][1].x = pt2.x;
            wider_lines[3][1].y = pt2.y;
        }
        else {
            true_lines[3][0].x = pt1.x;
            true_lines[3][0].y = pt1.y;
            true_lines[3][1].x = pt2.x;
            true_lines[3][1].y = pt2.y;
        }
    }
}


void Tray::addLine(bool vertical, Point p1, Point p2) {
    if (vertical) update_vertical_lines(p1, p2);
    else update_horizontal_lines(p1, p2);
}