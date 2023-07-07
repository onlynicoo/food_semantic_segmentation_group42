#include "../include/Test.h"
#include "../include/Tray.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>


/*
	Function that takes in input:
	x and y as rectangle's top left corner pixel coordinates
	height and corners as its measures
	centerX and centerY as references to save our calculations
*/
void calculateRectangleCentralPixel(int x, int y, int width, int height, int& centerX, int& centerY) {
	centerX = x + width / 2;
	centerY = y + height / 2;
}


/*
	Function that takes in input:
	groundTruthMask and ourMask

	This function returns the output as a pair:
	iou : iou calculated
*/
double singlePlateFoodSegmentation_IoUMetric(const cv::Mat& singleFoodMasks_prof, const cv::Mat& singleFoodMasks_our)
{
	double intersectionPixels = 0;
	double unionPixels = 0;

	for (int y = 0; y < singleFoodMasks_prof.rows; y++)
		for (int x = 0; x < singleFoodMasks_prof.cols; x++)
			if (singleFoodMasks_prof.at<uchar>(y, x) != 0)
			{
				unionPixels++;
				if (singleFoodMasks_our.at<uchar>(y, x) != 0)
					intersectionPixels++;

			}
			else
				if (singleFoodMasks_our.at<uchar>(y, x) != 0)
					unionPixels++;

	double IoU = (double)(intersectionPixels / unionPixels);
	return IoU;
}

/* 
	Function that takes in input:
	groundTruthMasks as masks grayScale image of test dataset 
	ourMasks as masks grayScale image calculated by us
	path_to_profBBs as path to test dataset file of bounding boxes
	path_to_ourBBs as path to our file of bounding boxes

	This function returns the output as a pair:
		iou : iou calculated in total
		numberFoodsGroundTruth: number of masks in groundTruthMasks
	
	mIoU will simply be iou/numberFoodsGroundTruth
*/
std::pair<double,int> OneImageSegmentation_IoUMetric(const cv::Mat& groundTruthMasks, const cv::Mat& ourMasks, std::string path_to_profBBs, std::string path_to_ourBBs)
{
	double iou = 0.0;
	int numberFoodsGroundTruth = 0;

	std::ifstream fileProf(path_to_profBBs);
	std::ifstream fileOur(path_to_ourBBs);

	std::string lineP, lineO;

	std::vector<std::pair<int, std::vector<int>>> rectanglesProf;
	std::vector<std::pair<int, std::vector<int>>> rectanglesOur;

	// Leggi il primo file e salva i rettangoli
	while (std::getline(fileProf, lineP)) {
		std::stringstream ssP(lineP);
		std::string token;
		std::getline(ssP, token, ':');
		int id = std::stoi(token);

		std::string coords;
		std::getline(ssP, coords, '[');
		std::getline(ssP, coords, ']');

		std::stringstream ssCoords(coords);
		std::string coordToken;

		std::vector<int> coordinates;
		while (std::getline(ssCoords, coordToken, ',')) {
			coordinates.push_back(std::stoi(coordToken));
		}

		if (coordinates.size() == 4) {
			rectanglesProf.push_back(std::make_pair(id, coordinates));
		}
	}

	// Leggi il secondo file e salva i rettangoli
	while (std::getline(fileOur, lineO)) {
		std::stringstream ss2(lineO);
		std::string token;
		std::getline(ss2, token, ':');
		int id = std::stoi(token);

		std::string coords;
		std::getline(ss2, coords, '[');
		std::getline(ss2, coords, ']');

		std::stringstream ssCoords(coords);
		std::string coordToken;

		std::vector<int> coordinates;
		while (std::getline(ssCoords, coordToken, ',')) {
			coordinates.push_back(std::stoi(coordToken));
		}

		if (coordinates.size() == 4) {
			rectanglesOur.push_back(std::make_pair(id, coordinates));
		}
	}

	// Confronta gli ID e calcola i centroidi corrispondenti
	for (const auto& rect1 : rectanglesProf) 
	{
		numberFoodsGroundTruth++;

		int centroid1X = -1, centroid1Y = -1;
		int centroid2X = -1, centroid2Y = -1;

		bool foundMatch = false;
		for (const auto& rect2 : rectanglesOur) {
			if (rect1.first == rect2.first) {
				foundMatch = true;

				//If true there's a match
				//I'll take the two gray levels representing the two masks gray levels
				//It will obvious to find in each BB's center, a pixel colored and not "a not mask" one

				const std::vector<int>& coords1 = rect1.second;
				int centroid1X, centroid1Y;
				calculateRectangleCentralPixel(coords1[0], coords1[1], coords1[2], coords1[3], centroid1X, centroid1Y);

				const std::vector<int>& coords2 = rect2.second;
				int centroid2X, centroid2Y;
				calculateRectangleCentralPixel(coords2[0], coords2[1], coords2[2], coords2[3], centroid2X, centroid2Y);

				break;
			}
		}

		if (!foundMatch) {
			//There's no match, no food item founded in our image (BAD NEWS)
			// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
		}

		if (foundMatch)
		{
			uchar grayProf = groundTruthMasks.at<uchar>(centroid1Y, centroid1X);
			uchar grayOur = ourMasks.at<uchar>(centroid2Y, centroid2X);

			cv::Mat oneMask_Prof(groundTruthMasks.size(), groundTruthMasks.type(),cv::Scalar(0));
			cv::Mat oneMask_Our(ourMasks.size(), ourMasks.type(), cv::Scalar(0));

			//Preparing the two matching masks
			for (int y = 0; y < groundTruthMasks.rows; y++)
				for (int x = 0; x < groundTruthMasks.cols; x++)
				{
					if (groundTruthMasks.at<uchar>(y, x) == grayProf) { oneMask_Prof.at<uchar>(y, x) = groundTruthMasks.at<uchar>(y, x); }
					if (ourMasks.at<uchar>(y, x) == grayOur) { oneMask_Our.at<uchar>(y, x) = ourMasks.at<uchar>(y, x); }
				}

			iou += singlePlateFoodSegmentation_IoUMetric(oneMask_Prof, oneMask_Our);
		}
	}

	fileProf.close();
	fileOur.close();
	
	return 	std::make_pair(iou, numberFoodsGroundTruth);
}



/*
	Function that takes in input:
	beforeMask and afterMask

	This function returns the Leftover Estimation metric: R_i
	R_i = #pixels in after image / #pixels in before image
*/
double singlePlateLeftoverEstimationMetric(const cv::Mat& beforeMask, const cv::Mat& afterMask)
{
	//beforeMask is mask of food before
	//afterMask is mask of food after
	int bMpixels = 0, aMpixels = 0;
	for (int y = 0; y < beforeMask.rows; y++)
		for (int x = 0; x < beforeMask.cols; x++)
			if (beforeMask.at<uchar>(y, x) != 0)
				bMpixels++;

	for (int y = 0; y < afterMask.rows; y++)
		for (int x = 0; x < afterMask.cols; x++)
			if (afterMask.at<uchar>(y, x) != 0)
				aMpixels++;

	double r_i = (double)(aMpixels / bMpixels);
	return r_i;
};



/*
	Function that takes in input:
	beforeImageMasks as masks grayScale image of test dataset
	afterImageMasks as masks grayScale image calculated by us
	beforeImageBB as path to test dataset file of bounding boxes
	AfterImageBB as path to our file of bounding boxes

	This function prints out:
	an ensemble of doubles. All the R_i's
*/
void OneImageSegmentation_LeftoverEstimation(const cv::Mat& beforeImageMasks, const cv::Mat& afterImageMasks, std::string beforeImageBB, std::string AfterImageBB)
{
	std::cout << "Taking: " + beforeImageBB + AfterImageBB + "\n";
	std::string output = "R_i's found: {   ";


	std::ifstream fileBef(beforeImageBB);
	std::ifstream fileAft(AfterImageBB);

	std::string lineBef, lineAft;

	std::vector<std::pair<int, std::vector<int>>> rectanglesBef;
	std::vector<std::pair<int, std::vector<int>>> rectanglesAft;

	// Leggi il primo file e salva i rettangoli
	while (std::getline(fileBef, lineBef)) {
		std::stringstream ssP(lineBef);
		std::string token;
		std::getline(ssP, token, ':');
		int id = std::stoi(token);

		std::string coords;
		std::getline(ssP, coords, '[');
		std::getline(ssP, coords, ']');

		std::stringstream ssCoords(coords);
		std::string coordToken;

		std::vector<int> coordinates;
		while (std::getline(ssCoords, coordToken, ',')) {
			coordinates.push_back(std::stoi(coordToken));
		}

		if (coordinates.size() == 4) {
			rectanglesBef.push_back(std::make_pair(id, coordinates));
		}
	}

	// Leggi il secondo file e salva i rettangoli
	while (std::getline(fileAft, lineAft)) {
		std::stringstream ss2(lineAft);
		std::string token;
		std::getline(ss2, token, ':');
		int id = std::stoi(token);

		std::string coords;
		std::getline(ss2, coords, '[');
		std::getline(ss2, coords, ']');

		std::stringstream ssCoords(coords);
		std::string coordToken;

		std::vector<int> coordinates;
		while (std::getline(ssCoords, coordToken, ',')) {
			coordinates.push_back(std::stoi(coordToken));
		}

		if (coordinates.size() == 4) {
			rectanglesAft.push_back(std::make_pair(id, coordinates));
		}
	}

	// Confronta gli ID e calcola i centroidi corrispondenti
	for (const auto& rect1 : rectanglesBef)
	{
		int centroid1X = -1, centroid1Y = -1;
		int centroid2X = -1, centroid2Y = -1;

		bool foundMatch = false;
		for (const auto& rect2 : rectanglesAft) {
			if (rect1.first == rect2.first) {
				foundMatch = true;

				//If true there's a match
				//I'll take the two gray levels representing the two masks gray levels
				//It will obvious to find in each BB's center, a pixel colored and not "a not mask" one

				const std::vector<int>& coords1 = rect1.second;
				int centroid1X, centroid1Y;
				calculateRectangleCentralPixel(coords1[0], coords1[1], coords1[2], coords1[3], centroid1X, centroid1Y);

				const std::vector<int>& coords2 = rect2.second;
				int centroid2X, centroid2Y;
				calculateRectangleCentralPixel(coords2[0], coords2[1], coords2[2], coords2[3], centroid2X, centroid2Y);

				break;
			}
		}

		if (!foundMatch) {
			//There's no match, no food item founded in leftover image
			//That means there's no food left
			output += "0  ";
		}

		if (foundMatch)
		{
			uchar grayBefore = beforeImageMasks.at<uchar>(centroid1Y, centroid1X);
			uchar grayAfter = afterImageMasks.at<uchar>(centroid2Y, centroid2X);

			cv::Mat oneMask_Before(beforeImageMasks.size(), beforeImageMasks.type(), cv::Scalar(0));
			cv::Mat oneMask_After(afterImageMasks.size(), afterImageMasks.type(), cv::Scalar(0));

			//Preparing the two matching masks
			for (int y = 0; y < beforeImageMasks.rows; y++)
				for (int x = 0; x < beforeImageMasks.cols; x++)
				{
					if (beforeImageMasks.at<uchar>(y, x) == grayBefore) { oneMask_Before.at<uchar>(y, x) = beforeImageMasks.at<uchar>(y, x); }
					if (afterImageMasks.at<uchar>(y, x) == grayAfter) { oneMask_After.at<uchar>(y, x) = afterImageMasks.at<uchar>(y, x); }
				}

			output += std::to_string(singlePlateLeftoverEstimationMetric(oneMask_Before, oneMask_After)) + "   ";
		}
	}

	fileBef.close();
	fileAft.close();

	output += "}";
	std::cout << output << "\n\n";
}



Test::Test(std::vector<Tray> vecTray) :trayVector{ vecTray } {}


int findTray(const std::string& str)
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

std::string leftover_or_foodimage(const std::string& str)
{
	//Hypothesis: "Food_leftover_dataset/tray1/x.jpg" or "tray1/x.jpg";

	// Trova la posizione dell'ultimo '/' nella stringa
	size_t lastSlashPos = str.find_last_of('/');

	std::string finalString;

	if (lastSlashPos != std::string::npos) 
	{
		// Estrai la sottostringa che segue l'ultimo '/'
		std::string extractedString = str.substr(lastSlashPos + 1);

		// Trova la posizione del primo '.' nella sottostringa
		size_t firstDotPos = extractedString.find_first_of('.');

		if (firstDotPos != std::string::npos) 
		{
			// Estrai la sottostringa che precede il primo '.'
			finalString = extractedString.substr(0, firstDotPos);
		}
	}

	return finalString;

}


void Test::test_the_system(const std::string& dataSetPath)
{
	std::vector<std::string> trayNames = { "food_image", "leftover1", "leftover2", "leftover3" };

	double mIoU = 0;
	double groundTruthFoods = 0;

	//We have our trayVector {(FOOD1/LEFTOVER1_F1),(FOOD1/LEFTOVER2_F1),(FOOD1/LEFTOVER3_F1),(FOOD2/LEFTOVER1_F2),...}
	for (int i = 0; i < trayVector.size(); i += 3)
	{
		//For general infos
		Tray t_lo1 = trayVector.at(i);
		//Indentifying Tray Number: elaborating it
		int TrayNumber = findTray(t_lo1.get_traysAfterNames());

		std::string prof_masks_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png";
		cv::Mat masks_fI; // = getMask(??);
		std::string bbRecords_fI; // = getBoundingBoxes(??);
		std::string profDataSet_bbRecord_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/food_image_bounding_box.txt";
		std::string profDataSet_masks_fI = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/food_image_mask.png";
		
		
		std::pair<double, int> result_of_single_IoU = OneImageSegmentation_IoUMetric(imread(prof_masks_fI,cv::IMREAD_GRAYSCALE), masks_fI, profDataSet_bbRecord_fI, bbRecords_fI);
		mIoU = (mIoU * groundTruthFoods + result_of_single_IoU.first) / result_of_single_IoU.second;

		/*
			We evaluate mIoU ONLY on leftovers 1 and 2
			We evaluate R_i ONLY on leftovers 1, 2 and 3
		*/
		for (int j = i; j - i < 3; j++)
		{
			//Leftover masks and bbs
			Tray t_LO = trayVector.at(j);
			cv::Mat masks_LO; // = getMask(t_LO);
			std::string boundingBoxes_LO; // = getBoundingBox Path

			//Corresponding prof's
			std::string profDataSet_mask_LO_path = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/masks/leftover/" + std::to_string(j+1) + ".png";
			std::string profDataSet_bbRecord_LO_path = dataSetPath + "/tray" + std::to_string(TrayNumber) + "/bounding_boxes/ " + std::to_string(j+1) + ".txt";
			cv::Mat profDataSet_mask_LO = imread(profDataSet_mask_LO_path, cv::IMREAD_GRAYSCALE);

			//We evaluate mIoU ONLY on leftovers 1 and 2
			if (j + 1 - i <= 2)
			{
				std::pair<double, int> result_of_single_IoU = OneImageSegmentation_IoUMetric(profDataSet_mask_LO, masks_LO, profDataSet_bbRecord_LO_path, boundingBoxes_LO);
				mIoU = (mIoU * groundTruthFoods + result_of_single_IoU.first) / result_of_single_IoU.second;

			}

			OneImageSegmentation_LeftoverEstimation(profDataSet_mask_LO, masks_LO, profDataSet_bbRecord_LO_path, boundingBoxes_LO);
		}

	}

}
