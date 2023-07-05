#include <opencv2/opencv.hpp>
#include "../include/Tray.h"


int main( int argc, char** argv ) {

    // Read arguments
    if (argc < 3) {
        std::cout << "You have to pass <initial tray image path> <final tray image path> as arguments." << std::endl;
        return 1;
    }
    std::string before = argv[1];
    std::string after = argv[2];
    
    Tray my_tray = Tray(before, after);
    
    my_tray.PrintInfo();
    
    return 0;
}
