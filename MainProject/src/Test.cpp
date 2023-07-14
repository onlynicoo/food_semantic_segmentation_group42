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


//Hypothesis: trayVector must contain samples like this:
//{(FOOD_IMAGE1/LEFTOVER1_F1),(FOOD_IMAGE1/LEFTOVER2_F1),(FOOD_IMAGE1/LEFTOVER3_F1),(FOOD_IMAGE2/LEFTOVER1_F2),...}
void Test::test_the_system(const std::string& dataSetPath)
{
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


	std::vector<double> mAPs;

	//We will examinate trayVector by multiples of 3
	for (int i = 0; i < trayVector.size(); i += 3)
	{
		//Indentifying Tray Number: elaborating it
		Tray t_lo1 = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo1.get_traysAfterNames());

		//Mettere directory output corretta. questa era di prova
		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourProjectMasks_fI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		std::string dataSetMasks_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png";

		//Food Image (before) bounding boxes
		std::string ourProjectBBRecords_fI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string dataSetBBRecords_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";

		std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations(0, imread(dataSetMasks_fI, cv::IMREAD_GRAYSCALE),
			dataSetBBRecords_fI, ourProjectMasks_fI, ourProjectBBRecords_fI, mAPs, groundTruthFoods);


		//Updating mIoU every time

		double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
		groundTruthFoods += result_of_single_IoU.second;
		mIoU = temp / groundTruthFoods;

		/*
		*   Now, leftover's examination
			We evaluate mIoU ONLY on leftovers 1 and 2
			We evaluate R_i ONLY on leftovers 1, 2 and 3
		*/
		for (int j = i; j - i < 3; j++)
		{
			//Leftover masks and bbs
			Tray t_loj = trayVector.at(j);
			cv::Mat ourProjectMasks_lj = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j % 3 + 1) + ".png", cv::IMREAD_GRAYSCALE);
			std::string ourProjectBBRecords_lj = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j % 3 + 1) + "_bounding_box.txt";

			//Corresponding prof's
			std::string dataSetMasks_lj = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j % 3 + 1) + ".png";
			std::string dataSetBBRecords_loj = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j % 3 + 1) + "_bounding_box.txt";

			std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations(j - i + 1, imread(dataSetMasks_lj, cv::IMREAD_GRAYSCALE), dataSetBBRecords_loj,
				ourProjectMasks_lj, ourProjectBBRecords_lj, mAPs, groundTruthFoods, imread(dataSetMasks_fI, cv::IMREAD_GRAYSCALE), dataSetBBRecords_fI);

			//Updating mIoU every time

			double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
			groundTruthFoods += result_of_single_IoU.second;
			mIoU = temp / groundTruthFoods;

		}

		numImagesDataset += 4;
	}

	double sumAP = 0.0;
	for (const auto& mAP : mAPs)
	{
		sumAP += mAP;
	}
	double mAP = sumAP / numImagesDataset;

	std::cout << "\nSystem AP : " << mAP << "\n";
	std::cout << "\nSystem mIoU : " << mIoU << "\n";
}
