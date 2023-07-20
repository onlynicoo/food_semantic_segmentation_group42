IMPORTANT!
Folder Structure:
In order to compile the project, create a "build" folder in the main directory "MainProject"
- makedir build
Enter the folder
- cd build
This two CMake command will compile the project
- cmake ..
- make

Inside build folder can be called main and mainTest
./main [...]
./mainText [...]

CMake will create in main directory output folder and a folder for each tray where the program will save masks and files.

LEGENDA:
	FLD  => food leftover's dataset
	mIoU => mean intersection over union
	mAP  => mean average precision
	LED  => leftover estimation difference

In this project there are two executables, with different purposes:

++++++++++++++++
+++ main.cpp +++
++++++++++++++++

Two different ways to make it run:

. With input parameters "before" := argv[1], "after" := argv[2]
	.. "before" is the path to one of the FLD's food images (before the meal).
	.. "after" is the path to one of the FLD's leftover images (difficulties 1, 2 or 3).
	.. This duo is used to create a Tray Object, so it is going to:

		__ Creates segmentation masks for before and after images.
		__ Creates bounding boxes files for before and after images.
		__ Saves them properly in ../output/trayX directory (with "X" from 1 to 8).

	.. In the end it will be shown all Tray's Infos (Before, Amount and Leftover 's amounts of pixels)
	.. Shown at video segmentation and localization's products.

. Without input parameters.
	.. It generates, all the proper possible combinations of food_image and leftover images.
	.. Every single combination is used to create a Tray Object, so it is going to:

		__ Create segmentation masks for before and after images.
		__ Create bounding boxes files for before and after images.
		__ Save them properly in ../output/trayX directory (with "X" from 1 to 8).

	.. In the end it will be shown all the Tray objects infos (Before, Amount and Leftover 's amounts of pixels).
	.. Shown at video segmentation and localization's products.

++++++++++++++++++++
+++ mainTest.cpp +++
++++++++++++++++++++

Two different ways to make it run.
. With input parameters "before" := argv[1], "after" := argv[2]
	.. "before" is the path to one of the FLD's food images (before the meal).
	.. "after" is the path to one of the FLD's leftover images (difficulties 1, 2 or 3).
	.. It generates a combination of food_image and leftover images, this duo is used to create a Tray Object, so it is going to:

		__ Create segmentation masks for before and after images.
		__ Create bounding boxes files for before and after images.
		__ Save them with properly in ../output/trayX directory (with "X" from 1 to 8).

	.. This Tray just created is put in a structure to make the function testTheSystem run.
	.. All performances' evaluation are done. Print out all mIoU, mAP and LED metrics results (of one Tray).

. Without input parameters:
	.. It generates, in order, all the proper possible combinations of food_image and leftover images
	.. Every single combination is used to create a Tray Object, so it is going to:

		__ Create segmentation masks for before and after images.
		__ Create bounding boxes files for before and after images.
		__ Save them with properly in ../output/trayX directory (with "X" from 1 to 8).

	.. All possible Tray objects are put in a vector of Tray objects, a proper structure to make the function testTheSystem run.
	.. All performances' evaluations are done. Print out all the mIoUs, mAPs and LEDs metrics results (of all wanted Trays).
