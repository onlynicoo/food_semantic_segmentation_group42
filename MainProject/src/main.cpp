#include <opencv2/opencv.hpp>
#include "../include/Tray.h"

 int main( int argc, char** argv ) {


     std::vector<Tray> trayVec;

     // Read arguments
     if (argc < 3) {
         for (int left = 3; left <= 3; left++) {
             for (int tray = 5; tray <= 8; tray++) {
                 std::cout << "Tray " + std::to_string(tray) << " Leftover " << std::to_string(left) << std::endl;
                 std::string str1 = "Food_leftover_dataset/tray" + std::to_string(tray) + "/food_image.jpg";
                 std::string str2 = "Food_leftover_dataset/tray" + std::to_string(tray) + "/leftover" + std::to_string(left) + ".jpg";
                 Tray my_tray = Tray(str1, str2);
                 trayVec.push_back(my_tray);
                 //my_tray.PrintInfo();
                 std::cout << std::endl;
             }
         }
     }
     else {
         std::string before = argv[1];
         std::string after = argv[2];
         Tray my_tray = Tray(before, after);
     }


     Test myTest = Test(trayVec);
     myTest.test_the_system("C:/Users/User/Desktop/prova finaleç/Food_leftover_dataset");

     return 0;
}