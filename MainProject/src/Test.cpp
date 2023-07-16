#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../include/Test.h"
#include "../include/Tray.h"
#include "../include/MetricsCalculation.h"


Test::Test(std::vector<Tray> vecTray) :trayVector{ vecTray } {}


int findTrayNumber(const std::string& str)
{
	std::istringstream iss(str);
	std::string token;

	int trayNumber = -1;  // Valore di default se il token non viene trovato

	while (iss >> token) {
		size_t found = token.find("tray");
		if (found != std::string::npos) {
			std::string trayNumStr = token.substr(found + 4);
			try {
				trayNumber = std::stoi(trayNumStr);
			}
			catch (const std::exception& e) {
				std::cout << "Errore: Impossibile convertire il numero di tray." << std::endl;
			}
			break;
		}
	}

	return trayNumber;
}

int findLeftoverNumber(const std::string& str)
{
	std::istringstream iss(str);
	std::string token;

	int leftNumber = -1;  // Valore di default se il token non viene trovato

	while (iss >> token) {
		size_t found = token.find("/leftover");
		if (found != std::string::npos) {
			std::string trayNumStr = token.substr(found + 9);
			try {
				leftNumber = std::stoi(trayNumStr);
			}
			catch (const std::exception& e) {
				std::cout << "Errore: Impossibile convertire il numero di tray." << std::endl;
			}
			break;
		}
	}

	return leftNumber;
}


//Hypothesis: trayVector must contain samples like this:
//{(FOOD_IMAGE1/LEFTOVER1_F1),(FOOD_IMAGE1/LEFTOVER2_F1),(FOOD_IMAGE1/LEFTOVER3_F1),(FOOD_IMAGE2/LEFTOVER1_F2),...}
void Test::test_the_system(const std::string& dataSetPath)
{

	std::cout << "---STARTING TEST---";

	//Each food segmented by us will be in this vector of Prediction. Look its definition
	std::vector<Prediction> predictions;
	//Each of 15 class we could predict from label 0 to label 14
	std::set<int> predictedClasses;
	//Mean intersection over union metric
	double mIoU = 0;
	//Number of each food found on professor's dataset
	int groundTruthFoods = 0;
	//Number of images on professor's dataset
	int numImagesDataset = 0;

	//FOR MAP
	std::vector<std::pair<int, int>> gTfoodItem_numbers;


	std::vector<double> mAPs;


	//We will examinate trayVector by multiples of 3
	for (int i = 0; i < trayVector.size(); i += 3)
	{
		//Indentifying Tray Number: elaborating it
		Tray t_lo1 = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo1.getTrayAfterPath());

		std::cout << "\n\n\nConsidering Tray: " << TrayNumber << "\n";
		std::cout << "FoodImage (Tray" << TrayNumber << ")\n\n";

		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourMasks_FI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		cv::Mat masks_FI = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);

		//Food Image (before) bounding boxes
		std::string ourBBs_FI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string gT_BBs_FI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";

		std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations_(0, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictions,
			predictedClasses, gTfoodItem_numbers,numImagesDataset, cv::Mat(), "", cv::Mat(), "");
		

		//Updating mIoU every time

		double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
		groundTruthFoods += result_of_single_IoU.second;
		mIoU = temp / groundTruthFoods;
		
		std::cout << "FoodImage (Tray " << TrayNumber << ") DONE\n\n\n";

		/*
		    Now, leftover's examination
			We evaluate mIoU ONLY on leftovers 1 and 2
			We evaluate R_i ONLY on leftovers 1, 2 and 3
		*/
		for (int j = i; j - i < 3; j++)
		{
			//Leftover masks and bbs
			Tray t_loj = trayVector.at(j);

			std::cout << "Leftover: " << j-i+1 << "(Tray " << TrayNumber << ")\n\n";

			cv::Mat ourMasks_LO = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j%3 + 1) + ".png", cv::IMREAD_GRAYSCALE);
			std::string ourBBs_LO = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j%3 + 1) + "_bounding_box.txt";

			//Corresponding prof's
			cv::Mat masks_LO = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j%3+1) + ".png", cv::IMREAD_GRAYSCALE);
			std::string gT_BBs_LO = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j%3+1) + "_bounding_box.txt";

			std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations_(j - i + 1, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictions, 
				predictedClasses, gTfoodItem_numbers, numImagesDataset, masks_LO, gT_BBs_LO, ourMasks_LO, ourBBs_LO);
				
			//Updating mIoU every time

			double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
			groundTruthFoods += result_of_single_IoU.second;
			mIoU = temp / groundTruthFoods;

			std::cout << "Leftover: " << j - i + 1 << "(Tray " << TrayNumber << ") DONE\n\n\n";
		}
	}

	double sumAP = 0.0;
	for (const auto& pc : predictedClasses)
	{
		int gtNumItemClass_pc = -1;
		for (const auto& nums : gTfoodItem_numbers)
		{
			if (nums.first == pc)
			{
				gtNumItemClass_pc = nums.second;
				break;
			}
		}
		sumAP += calculateAP(predictions, pc, gtNumItemClass_pc);
	}
	double mAP = sumAP / groundTruthFoods;

	std::cout << "mAP = " << mAP << "\n";
	std::cout << "mIoU = " << mIoU << "\n";
	std::cout << "---ENDING TEST---";
}




void Test::test_the_system_randomly(const std::string& dataSetPath)
{
	std::cout << "---STARTING TEST---";

	//Each food segmented by us will be in this vector of Prediction. Look its definition
	std::vector<Prediction> predictions;
	//Each of 15 class we could predict from label 0 to label 14
	std::set<int> predictedClasses;
	//Mean intersection over union metric
	double mIoU = 0;
	//Number of each food found on professor's dataset
	int groundTruthFoods = 0;
	//Number of images on professor's dataset
	int numImagesDataset = 0;

	//PAIRS FOR MAP
	std::vector<std::pair<int, int>> gTfoodItem_numbers;

	std::vector<double> mAPs;

	//We will examinate trayVector by multiples of 3
	for (int i = 0; i < trayVector.size(); i++)
	{
		//Indentifying Tray Number: elaborating it
		Tray t_lo = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo.getTrayAfterPath());
		int leftoverNumber = findLeftoverNumber(t_lo.getTrayAfterPath());

		std::cout << "\n\n\nConsidering Tray: " << TrayNumber << "\n";
		std::cout << "FoodImage (Tray" << TrayNumber << ")\n\n";

		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourMasks_FI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		cv::Mat masks_FI = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);

		//Food Image (before) bounding boxes
		std::string ourBBs_FI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string gT_BBs_FI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";


		std::pair<double, int> result_of_single_IoU_FI = OneImageSegmentation_MetricCalculations_(0, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictions,
			predictedClasses, gTfoodItem_numbers,numImagesDataset, cv::Mat(), "", cv::Mat(), "");

		//Updating mIoU every time

		double temp = mIoU * groundTruthFoods + result_of_single_IoU_FI.first;
		groundTruthFoods += result_of_single_IoU_FI.second;
		mIoU = temp / groundTruthFoods;

		std::cout << "FoodImage (Tray " << TrayNumber << ") DONE\n\n\n";

		std::cout << "Leftover: " << leftoverNumber << "(Tray " << TrayNumber << ")\n\n";

		cv::Mat ourMasks_LO = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string ourBBs_LO = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		//Corresponding prof's
		cv::Mat masks_LO = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string gT_BBs_LO = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		std::pair<double, int> result_of_single_IoU_LO = OneImageSegmentation_MetricCalculations_(leftoverNumber, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictions,
			predictedClasses, gTfoodItem_numbers, numImagesDataset, masks_LO, gT_BBs_LO, ourMasks_LO, ourBBs_LO);

		//Updating mIoU every time

		temp = mIoU * groundTruthFoods + result_of_single_IoU_LO.first;
		groundTruthFoods += result_of_single_IoU_LO.second;
		mIoU = temp / groundTruthFoods;

		std::cout << "Leftover: " << leftoverNumber << "(Tray " << TrayNumber << ") DONE\n\n\n";
	}

	double sumAP = 0.0;
	for (const auto& pc : predictedClasses)
	{
		int gtNumItemClass_pc = -1;
		for (const auto& nums : gTfoodItem_numbers)
		{
			if (nums.first == pc)
			{
				gtNumItemClass_pc = nums.second;
				break;
			}
		}

		sumAP += calculateAP(predictions, pc, gtNumItemClass_pc);
	}
	double mAP = sumAP / predictedClasses.size();

	std::cout << "mAP = " << mAP << "\n";
	std::cout << "mIoU = " << mIoU << "\n";
	std::cout << "---ENDING TEST---";
}