#include <opencv2/opencv.hpp>
#include "../include/Tray.h"

 int main( int argc, char** argv ) {

    // Read arguments
    if (argc < 3) {
        for (int left = 1; left <= 3; left++) {
            for (int tray = 1; tray <= 8; tray++) {
                std::cout << "Tray " + std::to_string(tray) << " Leftover " << std::to_string(left) << std::endl;
                Tray my_tray = Tray("../../../data/Food_leftover_dataset/tray" + std::to_string(tray) + "/food_image.jpg",
                                    "../../../data/Food_leftover_dataset/tray" + std::to_string(tray) + "/leftover" + std::to_string(left) + ".jpg");
                my_tray.PrintInfo();
                //my_tray.RefineSegmentation();
                std::cout << std::endl;
            }
        }
    } else {
        std::string before = argv[1];
        std::string after = argv[2];
        Tray my_tray = Tray(before, after);
        my_tray.PrintInfo();
        //my_tray.RefineSegmentation();
        std::cout << std::endl;
    }

    
    return 0;
}