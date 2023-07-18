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
 * The function `test_the_system' prints out all the info
 * about performance's metrics we've discussed in our report.
 * We take in input the "Food Leftover Dataset" by path, and we
 * use its masks and bounding boxes to calculate:
 * a) mAP of the system
 * b) mIoU of the system
 * c) leftover estimation
 * ASSUMPTION: Test::trayVector must contain samples like this:
 * {(F1/L1_F1),(F1/L2_F1),(F1/L3_F1),(F2/L1_F2),(F2/L2_F2),...}.
 *
 * @param dataSetPath path to "Food Leftover Dataset"
 */
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

	//Pairs (classID,#occurencies) in dataset
	std::vector<std::pair<int, int>> gTfoodItem_numbers;

	//All the matches found so far
	std::vector<double> mAPs;


	//We will examinate trayVector by multiples of 3
	for (int i = 0; i < trayVector.size(); i += 3)
	{

		std::vector<Prediction> predictionsOneTray;
		std::set<int> predictedClassesOneTray;
		std::vector<std::pair<int, int>> gTfoodItem_numbersOneTray;


		//Indentifying Tray Number: elaborating it
		Tray t_lo1 = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo1.getTrayAfterPath());

		std::cout << "\n\nFoodImage (Tray" << TrayNumber << ")\n";

		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourMasks_FI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		cv::Mat masks_FI = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);

		//Food Image (before) bounding boxes
		std::string ourBBs_FI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string gT_BBs_FI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";

		std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations_(0, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictionsOneTray,
			predictedClassesOneTray, gTfoodItem_numbersOneTray,numImagesDataset, cv::Mat(), "", cv::Mat(), "");
		

		//Updating mIoU every time
		double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
		groundTruthFoods += result_of_single_IoU.second;
		mIoU = temp / groundTruthFoods;
	

		/*
		    Now, leftover's examination
			We evaluate mIoU ONLY on leftovers 1 and 2
			We evaluate R_i ONLY on leftovers 1, 2 and 3
		*/
		for (int j = i; j - i < 3; j++)
		{

			if(j >= trayVector.size())
			{
				break;
			}

			std::vector<Prediction> predictionsOneTrayLeftover;
			std::set<int> predictedClassesOneTrayLeftover;
			std::vector<std::pair<int, int>> gTfoodItem_numbersOneTrayLeftover;



			//Leftover masks and bbs
			Tray t_loj = trayVector.at(j);

			std::cout << "\n\nLeftover: " << j-i+1 << "(Tray " << TrayNumber << ")\n";

			cv::Mat ourMasks_LO = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j%3 + 1) + ".png", cv::IMREAD_GRAYSCALE);
			std::string ourBBs_LO = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j%3 + 1) + "_bounding_box.txt";

			//Corresponding prof's
			cv::Mat masks_LO = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(j%3+1) + ".png", cv::IMREAD_GRAYSCALE);
			std::string gT_BBs_LO = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(j%3+1) + "_bounding_box.txt";

			std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations_(j - i + 1, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictionsOneTrayLeftover,
				predictedClassesOneTrayLeftover, gTfoodItem_numbersOneTrayLeftover, numImagesDataset, masks_LO, gT_BBs_LO, ourMasks_LO, ourBBs_LO);
				
			//Updating mIoU every time

			double temp = mIoU * groundTruthFoods + result_of_single_IoU.first;
			groundTruthFoods += result_of_single_IoU.second;
			mIoU = temp / groundTruthFoods;


			for (const auto& pot : predictionsOneTray)
				predictionsOneTrayLeftover.push_back(pot);

			for (const auto& pcot : predictedClassesOneTray)
				predictedClassesOneTrayLeftover.insert(pcot);

			for (int not_= 0; not_ < gTfoodItem_numbersOneTray.size(); not_++)
			{
				std::pair<int,int> not = gTfoodItem_numbersOneTray.at(not_);
				bool setted = false;
				for (int notl_ = 0; notl_ < gTfoodItem_numbersOneTrayLeftover.size(); notl_++)
				{
					std::pair<int, int> notl = gTfoodItem_numbersOneTrayLeftover.at(notl_);
					if (not.first == notl.first)
					{
						notl.second = notl.second + not.second;
						setted = true;
					}
				}
				if (!setted)
					gTfoodItem_numbersOneTrayLeftover.push_back(std::make_pair(not.first, not.second));
			}
			
			/*
			for (const auto & not : gTfoodItem_numbersOneTray)
			{
				bool setted = false;
				for (auto& notl : gTfoodItem_numbersOneTrayLeftover)
					if (not.first == notl.first)
					{
						notl.second = notl.second + not.second;
						setted = true;
					}
				if (!setted)
					gTfoodItem_numbersOneTrayLeftover.push_back(std::make_pair(not.first, not.second));
			}
			*/

			//Computing tray mAP
			double sumAPTrayLeftover = 0.0;
			for (const auto& pc_otl : predictedClassesOneTrayLeftover)
			{
				int gtNumItemClass_pcl = -1;
				for (const auto& nums_otl : gTfoodItem_numbersOneTrayLeftover)
				{
					if (nums_otl.first == pc_otl)
					{
						gtNumItemClass_pcl = nums_otl.second;
						break;
					}
				}

				sumAPTrayLeftover += calculateAP(predictionsOneTrayLeftover, pc_otl, gtNumItemClass_pcl);
			}
			double mAPOTL = sumAPTrayLeftover / predictedClassesOneTrayLeftover.size();

			std::cout << "Tray mAP = " << mAPOTL << "\n";
			std::cout << "\n\n";

			for (const auto& potl : predictionsOneTrayLeftover)
				predictions.push_back(potl);

			for (const auto& pcotl : predictedClassesOneTrayLeftover)
				predictedClasses.insert(pcotl);

			for (const auto & notl : gTfoodItem_numbersOneTrayLeftover)
			{
				bool setted = false;
				for (auto& n : gTfoodItem_numbers)
					if (notl.first == n.first)
					{
						n.second = n.second + notl.second;
						setted = true;
					}
				if (!setted)
					gTfoodItem_numbers.push_back(std::make_pair(notl.first, notl.second));
			}

			predictionsOneTrayLeftover.clear();
			predictedClassesOneTrayLeftover.clear();
			gTfoodItem_numbersOneTrayLeftover.clear();

		}

		predictionsOneTray.clear();
		predictedClassesOneTray.clear();
		gTfoodItem_numbersOneTray.clear();
	}

	//Computing overall mAP
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

	std::cout << "overall mAP = " << mAP << "\n";
	std::cout << "overall mIoU = " << mIoU << "\n";
	std::cout << "\n---ENDING TEST---";
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

	//Pairs (classID,#occurencies) in dataset
	std::vector<std::pair<int, int>> gTfoodItem_numbers;

	//All the matches found so far
	std::vector<double> mAPs;

	for (int i = 0; i < trayVector.size(); i++)
	{
		std::vector<Prediction> predictionsOneTray;
		std::set<int> predictedClassesOneTray;
		std::vector<std::pair<int, int>> gTfoodItem_numbersOneTray;


		//Indentifying Tray Number: elaborating it
		Tray t_lo = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo.getTrayAfterPath());
		int leftoverNumber = findLeftoverNumber(t_lo.getTrayAfterPath());

		std::cout << "\n\nFoodImage (Tray" << TrayNumber << ")\n";

		std::string outputPath = "../output";

		//Food Image (before) masks
		cv::Mat ourMasks_FI = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
		cv::Mat masks_FI = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);

		//Food Image (before) bounding boxes
		std::string ourBBs_FI = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string gT_BBs_FI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";


		std::pair<double, int> result_of_single_IoU_FI = OneImageSegmentation_MetricCalculations_(0, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictionsOneTray,
			predictedClassesOneTray, gTfoodItem_numbersOneTray,numImagesDataset, cv::Mat(), "", cv::Mat(), "");

		//Updating mIoU every time

		double temp = mIoU * groundTruthFoods + result_of_single_IoU_FI.first;
		groundTruthFoods += result_of_single_IoU_FI.second;
		mIoU = temp / groundTruthFoods;


		std::cout << "\n\nLeftover: " << leftoverNumber << "(Tray " << TrayNumber << ")\n";

		cv::Mat ourMasks_LO = cv::imread(outputPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string ourBBs_LO = outputPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		//Corresponding prof's
		cv::Mat masks_LO = cv::imread(dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover" + std::to_string(leftoverNumber) + ".png", cv::IMREAD_GRAYSCALE);
		std::string gT_BBs_LO = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/leftover" + std::to_string(leftoverNumber) + "_bounding_box.txt";

		std::pair<double, int> result_of_single_IoU_LO = OneImageSegmentation_MetricCalculations_(leftoverNumber, masks_FI, gT_BBs_FI, ourMasks_FI, ourBBs_FI, predictionsOneTray,
			predictedClassesOneTray, gTfoodItem_numbersOneTray, numImagesDataset, masks_LO, gT_BBs_LO, ourMasks_LO, ourBBs_LO);

		//Updating mIoU every time

		temp = mIoU * groundTruthFoods + result_of_single_IoU_LO.first;
		groundTruthFoods += result_of_single_IoU_LO.second;
		mIoU = temp / groundTruthFoods;


		//Computing tray mAP
		double sumAPTray = 0.0;
		for (const auto& pc_ot : predictedClassesOneTray)
		{
			int gtNumItemClass_pc = -1;
			for (const auto& nums_ot : gTfoodItem_numbersOneTray)
			{
				if (nums_ot.first == pc_ot)
				{
					gtNumItemClass_pc = nums_ot.second;
					break;
				}
			}

			sumAPTray += calculateAP(predictionsOneTray, pc_ot, gtNumItemClass_pc);
		}
		double mAPOT = sumAPTray / predictedClassesOneTray.size();

		for (const auto& pot : predictionsOneTray)
			predictions.push_back(pot);

		for (const auto& pcot : predictedClassesOneTray)
			predictedClasses.insert(pcot);


		for (int not_ = 0; not_ < gTfoodItem_numbersOneTray.size(); not_++)
		{
			std::pair<int,int> not = gTfoodItem_numbersOneTray.at(not_);
			bool setted = false;
			for (int n_ = 0; n_ < gTfoodItem_numbers.size(); n_++)
			{
				std::pair<int, int> n = gTfoodItem_numbers.at(n_);
				if (not.first == n.first)
				{
					n.second = n.second + not.second;
					setted = true;
				}
			}		
			if (!setted)
				gTfoodItem_numbers.push_back(std::make_pair(not.first, not.second));
		}

		/*
		for (const auto& not : gTfoodItem_numbersOneTray)
		{	
			bool setted = false;
			for (auto& n : gTfoodItem_numbers)
				if (not.first == n.first)
				{
					n.second = n.second + not.second;
					setted = true;
				}
			if (!setted)
				gTfoodItem_numbers.push_back(std::make_pair(not.first, not.second));
		}
		*/

		predictionsOneTray.clear();
		predictedClassesOneTray.clear();
		gTfoodItem_numbersOneTray.clear();


		std::cout << "Tray mAP = " << mAPOT << "\n";
		std::cout << "\n\n";
	}

	//Computing overall mAP
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

	std::cout << "overall mAP = " << mAP << "\n";
	std::cout << "overall mIoU = " << mIoU << "\n";
	std::cout << "\n---ENDING TEST---";
}