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

	while (iss >> token)
	{
		if (token.find("tray") != std::string::npos) 
		{
			// Trovato il token "tray", estrai il numero successivo
			std::string trayNumStr = token.substr(4);  // Ignora i primi 4 caratteri ("tray")
			try {
				trayNumber = std::stoi(trayNumStr);
			}
			catch (const std::exception& e) {
				// Errore di conversione del numero
				std::cout << "Errore: Impossibile convertire il numero di tray." << std::endl;
			}
			break;  // Esci dal ciclo
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
	double groundTruthFoods = 0;


	//We will examinate trayVector by multiples of 3
	for (int i = 0; i < trayVector.size(); i += 3)
	{
		//Indentifying Tray Number: elaborating it
		Tray t_lo1 = trayVector.at(i);
		int TrayNumber = findTrayNumber(t_lo1.get_traysAfterNames());

		//Food Image (before) masks
		cv::Mat ourProjectMasks_fI; // = getMask(??);
		std::string dataSetMasks_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png";

		//Food Image (before) bounding boxes
		std::string ourProjectBBRecords_fI; // = getBoundingBoxes(??);
		std::string dataSetBBRecords_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		
	
		/*std::pair<double, int> result_of_single_IoU = OneImageSegmentation_IoUMetric(
			imread(dataSetMasks_fI,cv::IMREAD_GRAYSCALE), ourProjectMasks_fI, dataSetBBRecords_fI,
			ourProjectBBRecords_fI, predictions,predictedClasses);*/

		std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations(0, imread(dataSetMasks_fI,cv::IMREAD_GRAYSCALE),
			dataSetBBRecords_fI,ourProjectMasks_fI, dataSetBBRecords_fI, predictions,predictedClasses);
		
		//Updating mIoU every time
		mIoU = (mIoU * groundTruthFoods + result_of_single_IoU.first) / result_of_single_IoU.second;


		/*
		*   Now, leftover's examination
			We evaluate mIoU ONLY on leftovers 1 and 2
			We evaluate R_i ONLY on leftovers 1, 2 and 3
		*/
		for (int j = i; j - i < 3; j++)
		{
			//Leftover masks and bbs
			Tray t_loj = trayVector.at(j);
			cv::Mat ourProjectMasks_lj; // = getMask(t_LO);
			std::string ourProjectBBRecords_lj; // = getBoundingBox Path

			//Corresponding prof's
			std::string dataSetMasks_lj = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover/" + std::to_string(j+1) + ".png";
			std::string dataSetBBRecords_loj = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/ " + std::to_string(j+1) + ".txt";

			std::pair<double, int> result_of_single_IoU = OneImageSegmentation_MetricCalculations(j - i + 1, imread(dataSetMasks_lj, cv::IMREAD_GRAYSCALE), dataSetBBRecords_loj,
				ourProjectMasks_lj, ourProjectBBRecords_lj, predictions, predictedClasses, imread(dataSetMasks_fI, cv::IMREAD_GRAYSCALE), dataSetBBRecords_fI);
				
			//Updating mIoU every time
			mIoU = (mIoU * groundTruthFoods + result_of_single_IoU.first) / result_of_single_IoU.second;

		}

	}

	double sumAP=0.0;
	for (const auto& c : predictedClasses)
	{
		sumAP += calculateAP(predictions,(int)c);
	}
	double mAP = sumAP / predictedClasses.size();

	std::cout << "\nSystem AP : " << mAP << "\n";
	std::cout << "\nSystem mIoU : " << mIoU << "\n";
}
