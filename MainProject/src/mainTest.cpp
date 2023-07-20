#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "../include/Tray.h"
#include "../include/Test.h"

int main( int argc, char** argv ) {


    std::vector<Tray> trayVec;

    // Read arguments
    if (argc < 3) {
        for (int left = 1; left <= 1; left++) {
            for (int tray = 1; tray <= 1; tray++) {
            std::cout << "Tray " + std::to_string(tray) << " Leftover " << std::to_string(left) << std::endl;
            std::string str1 = "../input/Food_leftover_dataset/tray" + std::to_string(tray) + "/food_image.jpg";
            std::string str2 = "../input/Food_leftover_dataset/tray" + std::to_string(tray) + "/leftover" + std::to_string(left) + ".jpg";
            Tray my_tray = Tray(str1, str2);
            trayVec.push_back(my_tray);
            // my_tray.PrintInfo();
            std::cout << std::endl;
            }
        }
    }
    else {
        std::string before = argv[1];
        std::string after = argv[2];
        Tray my_tray = Tray(before, after);
        trayVec.push_back(my_tray);
    }
    Test myTestR = Test(trayVec);
    myTestR.test_the_system("../input/Food_leftover_dataset");


    return 0;
}