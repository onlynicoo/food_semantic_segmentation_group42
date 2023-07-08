#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../include/Test.h"
#include "../include/Tray.h"

/*
	Class useful to calculate mAP
	Each Prediction object has:
		- Identified class
		- Confidence score
		- An object is a TP iff confidence score > 0.50
*/
class Prediction {

	private:
		int classID;
		double confidence;
		bool isTruePositive;

	public:

		int getClassId() const { return classID; }
		double getConfidence() const { return confidence; }
		bool isTP() const { return isTruePositive; }
	
		Prediction(int cid, double conf, bool iTP)
			: classID(cid), confidence(conf), isTruePositive(iTP) {}
};


/*
	It calculates the Precision-Recall curve.
	Also it gives us the AP for one single class of objects

	For more info : https: // learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
					https: //towardsdatascience.com/a-better-map-for-object-detection-32662767d424
*/
double calculateAP(const std::vector<Prediction>& predictions, int classID) {
	int truePositives = 0;
	int falsePositives = 0;

	// Ordina le predizioni in ordine decrescente di confidenza
	std::vector<Prediction> sortedPredictions = predictions;
	std::sort(sortedPredictions.begin(), sortedPredictions.end(), [](const Prediction& a, const Prediction& b) {
		return a.getConfidence() > b.getConfidence();
		});

	// Calcola i valori di precisione e recall per ogni predizione
	std::vector<double> precision;
	std::vector<double> recall;
	for (const Prediction& pred : sortedPredictions) {
		if (pred.getClassId() == classID) {
			if (pred.isTP()) {
				truePositives++;
			}
			else {
				falsePositives++;
			}
		}

		double currPrecision = static_cast<double>(truePositives) / (truePositives + falsePositives);
		double currRecall = static_cast<double>(truePositives) / truePositives; // Recall è sempre 1

		precision.push_back(currPrecision);
		recall.push_back(currRecall);
	}

	// Calcola l'Average Precision (AP) utilizzando l'interpolazione a integrali
	double averagePrecision = 0.0;
	double prevRecall = 0.0;
	for (size_t i = 0; i < precision.size(); i++) {
		if (recall[i] != prevRecall) {
			averagePrecision += precision[i] * (recall[i] - prevRecall);
			prevRecall = recall[i];
		}
	}

	return averagePrecision;
}




/*	
	Structure useful for our Metrics
	It takes into account the 
	"ID: id; [x, y, width, heigth]" format
	
	Prof means that the BB was taken from
	"The best prof" dataset

*/
class RectangleFileProf
{
	private: 
		int rectID;
		std::vector<int> coordinates;

	public:

		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; } 

		RectangleFileProf(int id, const std::vector<int>& coords)
			: rectID(id), coordinates(coords) {}
};




/*
	Structure useful for our Metrics
	It takes into account the
	"ID: id; [x, y, width, heigth]" format

	Our means that BB was taken from
	our detected bounding boxes

	ALSO:
		isTaken: as a bool to see if a BB(refering to a detected food) has
				a match in the profdataset file of BB

		prediction: as the metric of confidence score of that detection, useful
					for calculate mAP

*/
class RectangleFileOur
{
	private:
		int rectID;
		std::vector<int> coordinates;
		bool isTaken;
		double prediction;

	public:

		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; }
		bool getIsTaken() const { return isTaken; }
		double getPrediction() const { return prediction; }

		void setIsTaken(bool taken) { isTaken = taken; } 
		void setPrediction(bool pred) { prediction = pred; }

		RectangleFileOur(int id, const std::vector<int>& coords, bool tak)
			: rectID(id), coordinates(coords), isTaken(tak), prediction(0){}
};




/*
	Function that takes in input:
	x and y as BB's top left corner pixel coordinates
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



std::pair<std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string profBB_path, std::string ourBB_path)
{
	std::ifstream fileProf(profBB_path);
	std::ifstream fileOur(ourBB_path);

	std::string lineP, lineO;

	std::vector<RectangleFileProf> rectanglesProf;
	std::vector<RectangleFileOur> rectanglesOur;

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
			rectanglesProf.push_back(RectangleFileProf(id, coordinates));
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
			rectanglesOur.push_back(RectangleFileOur(id, coordinates, false));
		}
	}

	return std::make_pair(rectanglesProf, rectanglesOur);

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
std::pair<double,int> OneImageSegmentation_IoUMetric(const cv::Mat& groundTruthMasks, const cv::Mat& ourMasks, std::string path_to_profBBs, std::string path_to_ourBBs, std::vector<Prediction>&predictions, std::set<int> &predicted)
{
	double iou = 0.0;
	int numberFoodsGroundTruth = 0;

	std::ifstream fileProf(path_to_profBBs);
	std::ifstream fileOur(path_to_ourBBs);

	std::string lineP, lineO;

	std::vector<RectangleFileProf> rectanglesProf;
	std::vector<RectangleFileOur> rectanglesOur;

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
			rectanglesProf.push_back(RectangleFileProf(id,coordinates));
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
			rectanglesOur.push_back(RectangleFileOur(id,coordinates,false));
		}
	}

	// Confronta gli ID e calcola i centroidi corrispondenti
	for (const auto& rect1 : rectanglesProf) 
	{
		numberFoodsGroundTruth++;

		int centroid1X = -1, centroid1Y = -1;
		int centroid2X = -1, centroid2Y = -1;

		bool foundMatch = false;

		for (auto& rect2 : rectanglesOur) {
			if (rect1.getRectId() == rect2.getRectId()) {
				foundMatch = true;
				rect2.setIsTaken(true);
				//If true there's a match
				//I'll take the two gray levels representing the two masks gray levels
				//It will obvious to find in each BB's center, a pixel colored and not "a not mask" one

				const std::vector<int>& coords1 = rect1.getCoords();
				int centroid1X, centroid1Y;
				calculateRectangleCentralPixel(coords1[0], coords1[1], coords1[2], coords1[3], centroid1X, centroid1Y);

				const std::vector<int>& coords2 = rect2.getCoords();
				int centroid2X, centroid2Y;
				calculateRectangleCentralPixel(coords2[0], coords2[1], coords2[2], coords2[3], centroid2X, centroid2Y);


				//If found!

				uchar grayProf = groundTruthMasks.at<uchar>(centroid1Y, centroid1X);
				uchar grayOur = ourMasks.at<uchar>(centroid2Y, centroid2X);

				cv::Mat oneMask_Prof(groundTruthMasks.size(), groundTruthMasks.type(), cv::Scalar(0));
				cv::Mat oneMask_Our(ourMasks.size(), ourMasks.type(), cv::Scalar(0));

				//Preparing the two matching masks
				for (int y = 0; y < groundTruthMasks.rows; y++)
					for (int x = 0; x < groundTruthMasks.cols; x++)
					{
						if (groundTruthMasks.at<uchar>(y, x) == grayProf) { oneMask_Prof.at<uchar>(y, x) = groundTruthMasks.at<uchar>(y, x); }
						if (ourMasks.at<uchar>(y, x) == grayOur) { oneMask_Our.at<uchar>(y, x) = ourMasks.at<uchar>(y, x); }
					}

				double single_iou = singlePlateFoodSegmentation_IoUMetric(oneMask_Prof, oneMask_Our);
				rect2.setPrediction(single_iou);
				iou += single_iou;

				break;
			}
		}

		if (!foundMatch) {
			//There's no match, no food item founded in our image (BAD NEWS)
			// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
		}

	}

	fileProf.close();
	fileOur.close();
	

	for (const auto& r : rectanglesOur)
	{
		int id = r.getRectId();
		predicted.insert(id);
		bool isTruePositive = false;
		if (r.getPrediction() > 0.50) isTruePositive = true;
		predictions.push_back(Prediction(id,r.getPrediction(),isTruePositive));
	}


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











std::pair<double, int> OneImageSegmentation_MetricCalculations(
	int code,
	const cv::Mat& gT_Masks,
	const std::string gT_BBs,
	const cv::Mat& ourMasks,
	std::string ourBBs,
	std::vector<Prediction>& predictions, 
	std::set<int>& predicted,
	const cv::Mat& beforeMasks = cv::Mat(), /* gT_FI_masks */
	const std::string beforeBBs = std::string() /* gT_FI_BB */)
{
	double iou = 0.0;
	int numberFoodsGroundTruth = 0;

	std::cout << "Taking: " + beforeBBs + " and " +  ourBBs + "\n";
	std::string output = "R_i's found: {   ";

	std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gT_BBs, ourBBs);
	std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForLE = boundingBoxFileTokenizer(beforeBBs, ourBBs);
	std::vector<RectangleFileProf> gT_rects = rectsForIoU.first;
	std::vector<RectangleFileOur> our_rects = rectsForIoU.second;
	std::vector<RectangleFileProf> before_rects = rectsForLE.first;
	std::vector<RectangleFileOur> after_rects = rectsForLE.second;

	// Confronteremo gli ID e calcoleremo i centroidi corrispondenti

	//IoU ESTIMATION PART
	if (code == 0 || code == 1 || code == 2)
	{
		for (const auto& gT_rect : gT_rects)
		{
			numberFoodsGroundTruth++;

			int centroid1X = -1, centroid1Y = -1;
			int centroid2X = -1, centroid2Y = -1;

			bool foundMatch = false;

			for (auto& our_rect : our_rects) {
				if (gT_rect.getRectId() == our_rect.getRectId()) {
					foundMatch = true;
					our_rect.setIsTaken(true);
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					//It will obvious to find in each BB's center, a pixel colored and not "a not mask" one

					const std::vector<int>& coords1 = gT_rect.getCoords();
					int centroid1X, centroid1Y;
					calculateRectangleCentralPixel(coords1[0], coords1[1], coords1[2], coords1[3], centroid1X, centroid1Y);

					const std::vector<int>& coords2 = our_rect.getCoords();
					int centroid2X, centroid2Y;
					calculateRectangleCentralPixel(coords2[0], coords2[1], coords2[2], coords2[3], centroid2X, centroid2Y);


					//If found!

					uchar grayProf = gT_Masks.at<uchar>(centroid1Y, centroid1X);
					uchar grayOur = ourMasks.at<uchar>(centroid2Y, centroid2X);

					cv::Mat oneMask_Prof(gT_Masks.size(), gT_Masks.type(), cv::Scalar(0));
					cv::Mat oneMask_Our(ourMasks.size(), ourMasks.type(), cv::Scalar(0));

					//Preparing the two matching masks
					for (int y = 0; y < gT_Masks.rows; y++)
						for (int x = 0; x < gT_Masks.cols; x++)
						{
							if (gT_Masks.at<uchar>(y, x) == grayProf) { oneMask_Prof.at<uchar>(y, x) = gT_Masks.at<uchar>(y, x); }
							if (ourMasks.at<uchar>(y, x) == grayOur) { oneMask_Our.at<uchar>(y, x) = ourMasks.at<uchar>(y, x); }
						}


					double single_iou = singlePlateFoodSegmentation_IoUMetric(oneMask_Prof, oneMask_Our);
					our_rect.setPrediction(single_iou);
					iou += single_iou;
					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our image (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
			}

		}

		for (const auto& r : our_rects)
		{
			int id = r.getRectId();
			predicted.insert(id);
			bool isTruePositive = false;
			if (r.getPrediction() > 0.50) isTruePositive = true;
			predictions.push_back(Prediction(id, r.getPrediction(), isTruePositive));
		}
	}

	//LEFTOVER ESTIMATION PART
	if (code == 1 || code == 2 || code == 3)
	{
		for (const auto& before_rect : before_rects)
		{
			int centroid1X = -1, centroid1Y = -1;
			int centroid2X = -1, centroid2Y = -1;

			bool foundMatch = false;

			for (auto& after_rect : after_rects) {
				if (before_rect.getRectId() == after_rect.getRectId()) {
					foundMatch = true;
					after_rect.setIsTaken(true);
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					//It will obvious to find in each BB's center, a pixel colored and not "a not mask" one

					const std::vector<int>& coords1 = before_rect.getCoords();
					int centroid1X, centroid1Y;
					calculateRectangleCentralPixel(coords1[0], coords1[1], coords1[2], coords1[3], centroid1X, centroid1Y);

					const std::vector<int>& coords2 = after_rect.getCoords();
					int centroid2X, centroid2Y;
					calculateRectangleCentralPixel(coords2[0], coords2[1], coords2[2], coords2[3], centroid2X, centroid2Y);


					//If found!

					uchar grayBefore = beforeMasks.at<uchar>(centroid1Y, centroid1X);
					uchar grayAfter = ourMasks.at<uchar>(centroid2Y, centroid2X);

					cv::Mat oneMask_Before(beforeMasks.size(), beforeMasks.type(), cv::Scalar(0));
					cv::Mat oneMask_After(ourMasks.size(), ourMasks.type(), cv::Scalar(0));

					//Preparing the two matching masks
					for (int y = 0; y < beforeMasks.rows; y++)
						for (int x = 0; x < beforeMasks.cols; x++)
						{
							if (beforeMasks.at<uchar>(y, x) == grayBefore) { oneMask_Before.at<uchar>(y, x) = beforeMasks.at<uchar>(y, x); }
							if (ourMasks.at<uchar>(y, x) == grayAfter) { oneMask_After.at<uchar>(y, x) = ourMasks.at<uchar>(y, x); }
						}

					output += std::to_string(singlePlateLeftoverEstimationMetric(oneMask_Before, oneMask_After)) + "   ";
					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our image (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				output += "0  ";
			}

		}

		output += "}";
		std::cout << output << "\n\n";
	}

	
	return 	std::make_pair(iou, numberFoodsGroundTruth);


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
		sumAP += calculateAP(predictions,c);
	}
	double mAP = sumAP / predictedClasses.size();

	std::cout << "\nSystem AP : " << mAP << "\n";
	std::cout << "\nSystem mIoU : " << mIoU << "\n";
}
