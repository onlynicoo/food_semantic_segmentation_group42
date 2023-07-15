#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/Tray.h"

int main(int argc, char** argv) {
    std::vector<Tray> trayVec;

    // Read arguments
    if (argc < 3) {
        for (int left = 1; left <= 3; left++) {
            for (int tray = 1; tray <= 8; tray++) {
            std::cout << "Tray " + std::to_string(tray) << " Leftover " << std::to_string(left) << std::endl;
            std::string str1 = "../input/Food_leftover_dataset/tray" + std::to_string(tray) + "/food_image.jpg";
            std::string str2 = "../input/Food_leftover_dataset/tray" + std::to_string(tray) + "/leftover" + std::to_string(left) + ".jpg";
            Tray my_tray = Tray(str1, str2);
            trayVec.push_back(my_tray);
            my_tray.showTray();
            std::cout << std::endl;
            }
        }
    } else {
        std::string before = argv[1];
        std::string after = argv[2];
        Tray my_tray = Tray(before, after);
        my_tray.showTray();
        //my_tray.RefineSegmentation();
        
    }
    return 0;
}