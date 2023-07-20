#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "../include/Test.h"
#include "../include/Tray.h"
#include "../include/MetricsCalculation.h"

/**
 * The Test constructor takes in input one single parameter: a stl
 * vector of Tray objects. Within this vector, we will have all 
 * possible info for all the metrics we need to evaluate our
 * system performances. We will scan it (in several ways) to retrieve
 * all food image and leftovers bounding box and masks infos and compare
 * them to Ground Truth.
 * 
 * @param vecTray stl vector containing all Tray obects, previously computed
 */
Test::Test(std::vector<Tray> vecTray) :trayVector{ vecTray } {}

/**
 * The function `findTrayNumber' gives us an integer representing
 * the tray number we're working on
 *
 * @param str path/string for a bounding box txt file
 * @return an integer: the tray number
 */
int findTrayNumber(const std::string& str)
{
	std::istringstream iss(str);
	std::string token;

	int trayNumber = -1;  // Default value. Returned if nothing found

	while (iss >> token) {
		size_t found = token.find("tray");
		if (found != std::string::npos) {
			std::string trayNumStr = token.substr(found + 4);
			try {
				trayNumber = std::stoi(trayNumStr);
			}
			catch (const std::exception& e) {
				std::cout << "Error: Impossible to convert tray number just found." << std::endl;
			}
			break;
		}
	}

	return trayNumber;
}

/**
 * The function `findLeftoverNumber' gives us an integer representing
 * the leftover number we're working on
 *
 * @param str path/string for a bounding box txt file
 * @return an integer: the leftover number
 */
int findLeftoverNumber(const std::string& str)
{
	std::istringstream iss(str);
	std::string token;

	int leftNumber = -1;  // Default value. Returned if nothing found

	while (iss >> token) {
		size_t found = token.find("/leftover");
		if (found != std::string::npos) {
			std::string trayNumStr = token.substr(found + 9);
			try {
				leftNumber = std::stoi(trayNumStr);
			}
			catch (const std::exception& e) {
				std::cout << "Error: Impossible to convert leftover number just found." << std::endl;
			}
			break;
		}
	}

	return leftNumber;
}

/**
 * The function `test_the_system_randomly' prints out all the info
 * about performance's metrics we've discussed in our report.
 * We take in input the "Food Leftover Dataset" by path, and we
 * use its masks and bounding boxes to calculate:
 * a) mAP of the system
 * b) mIoU of the system
 * c) leftover estimation
 * ASSUMPTION: trayVector can contain each pair (FX/LY_X) in every order.
 *
 * @param dataSetPath path to "Food Leftover Dataset"
 */
void Test::testTheSystem(const std::string& dataSetPath)
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

	//Pairs (classID,#occurencies) in dataset
	std::vector<std::pair<int, int>> gTfoodItemNumbers;

	//All the matches found so far
	std::vector<double> mAPs;

	for (int i = 0; i < trayVector.size(); i++)
	{
		std::vector<Prediction> predictionsOneTray;
		std::set<int> predictedClassesOneTray;
		std::vector<std::pair<int, int>> gTfoodItemNumbersOneTray;


		//Indentifying Tray Number: elaborating it
		Tray tLo = trayVector.at(i);
		int TrayNumber = findTrayNumber(tLo.getTrayAfterPath());
		int leftoverNumber = findLeftoverNumber(tLo.getTrayAfterPath());

		std::cout << "\n\nFoodImage (Tray" << TrayNumber << ")\n";

		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourMasksFI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		cv::Mat masksFI = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);

		//Food Image (before) bounding boxes
		std::string ourBBsFI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string gTBBsFI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";


		std::pair<double, int> resultOfSingleIoUFI = OneImageSegmentationMetricCalculations(0, masksFI, gTBBsFI, ourMasksFI, ourBBsFI, predictionsOneTray,
			predictedClassesOneTray, gTfoodItemNumbersOneTray,numImagesDataset, cv::Mat(), "", cv::Mat(), "");

		//Updating mIoU every time
		double temp = mIoU * groundTruthFoods + resultOfSingleIoUFI.first;
		groundTruthFoods += resultOfSingleIoUFI.second;
		mIoU = temp / groundTruthFoods;


		//Computing food_image mAP
		double sumAPFI = 0.0;
		for (const auto& pcFI : predictedClassesOneTray)
		{
			int gtNumItemClassPC = -1;
			for (const auto& nums_ot : gTfoodItemNumbersOneTray)
			{
				if (nums_ot.first == pcFI)
				{
					gtNumItemClassPC = nums_ot.second;
					break;
				}
			}

			sumAPFI += calculateAP(predictionsOneTray, pcFI, gtNumItemClassPC);
		}
		double mAPFI = sumAPFI / predictedClassesOneTray.size();
		std::cout << "\nFood Image mAP = " << mAPFI << "\n\n";


		std::cout << "\n\nLeftover: " << leftoverNumber << "(Tray " << TrayNumber << ")\n";

		std::vector<Prediction> predictionsLeftover;
		std::set<int> predictedClassesLeftover;
		std::vector<std::pair<int, int>> gTfoodItemNumbersLeftover;

		cv::Mat ourMasksLO = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string ourBBsLO = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		//Corresponding prof's
		cv::Mat masksLO = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string gTBBsLO = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		std::pair<double, int> resultOfSingleIoULO = OneImageSegmentationMetricCalculations(leftoverNumber, masksFI, gTBBsFI, ourMasksFI, ourBBsFI, predictionsLeftover,
			predictedClassesLeftover, gTfoodItemNumbersLeftover, numImagesDataset, masksLO, gTBBsLO, ourMasksLO, ourBBsLO);

		//Updating mIoU every time

		temp = mIoU * groundTruthFoods + resultOfSingleIoULO.first;
		groundTruthFoods += resultOfSingleIoULO.second;
		mIoU = temp / groundTruthFoods;

		if (leftoverNumber != 3)
		{//Computing leftover mAP
			double sumAPLO = 0.0;
			for (const auto& pcLO : predictedClassesLeftover)
			{
				int gtNumItemClassPCLO = -1;
				for (const auto& numsOt : gTfoodItemNumbersLeftover)
				{
					if (numsOt.first == pcLO)
					{
						gtNumItemClassPCLO = numsOt.second;
						break;
					}
				}

				sumAPLO += calculateAP(predictionsLeftover, pcLO, gtNumItemClassPCLO);
			}
			double mAPLO = sumAPLO / predictedClassesLeftover.size();
			std::cout << "Leftover mAP = " << mAPLO << "\n\n";
		}


		for (const auto& polo : predictionsLeftover)
			predictionsOneTray.push_back(polo);

		for (const auto& pclo : predictedClassesLeftover)
			predictedClassesOneTray.insert(pclo);


		for (const auto& nLO : gTfoodItemNumbersLeftover)
		{
			bool setted = false;
			for (auto& nOt : gTfoodItemNumbersOneTray)
			{
				if (nLO.first == nOt.first)
				{
					nOt.second = nOt.second + nLO.second;
					setted = true;
				}
			}
			if (!setted)
				gTfoodItemNumbersOneTray.push_back(std::make_pair(nLO.first, nLO.second));
		}


		//Computing tray mAP
		double sumAPTray = 0.0;
		for (const auto& pcOt : predictedClassesOneTray)
		{
			int gtNumItemClass_pc = -1;
			for (const auto& numsOt : gTfoodItemNumbersOneTray)
			{
				if (numsOt.first == pcOt)
				{
					gtNumItemClass_pc = numsOt.second;
					break;
				}
			}

			sumAPTray += calculateAP(predictionsOneTray, pcOt, gtNumItemClass_pc);
		}
		double mAPOT = sumAPTray / predictedClassesOneTray.size();
		std::cout << "\nTray mAP = " << mAPOT << "\n";


		//Put One Tray stuff in the overall structures
		for (const auto& pot : predictionsOneTray)
			predictions.push_back(pot);
		for (const auto& pcot : predictedClassesOneTray)
			predictedClasses.insert(pcot);
		for (const auto& nOt : gTfoodItemNumbersOneTray)
		{
			bool setted = false;
			for (auto& n : gTfoodItemNumbers)
			{
				if (nOt.first == n.first)
				{
					n.second = n.second + nOt.second;
					setted = true;
				}

			}
			if (!setted)
				gTfoodItemNumbers.push_back(std::make_pair(nOt.first, nOt.second));
		}

		predictionsOneTray.clear();
		predictedClassesOneTray.clear();
		gTfoodItemNumbersOneTray.clear();
		predictionsLeftover.clear();
		predictedClassesLeftover.clear();
		gTfoodItemNumbersLeftover.clear();

		std::cout << "\n\n";
	}

	//Computing overall mAP
	double sumAP = 0.0;
	for (const auto& pc : predictedClasses)
	{
		int gtNumItemClassPC = -1;
		for (const auto& nums : gTfoodItemNumbers)
		{
			if (nums.first == pc)
			{
				gtNumItemClassPC = nums.second;
				break;
			}
		}


		sumAP += calculateAP(predictions, pc, gtNumItemClassPC);
	}
	double mAP = sumAP / predictedClasses.size();

	std::cout << "overall mAP = " << mAP << "\n";
	std::cout << "overall mIoU = " << mIoU << "\n";
	std::cout << "\n---ENDING TEST---";
}