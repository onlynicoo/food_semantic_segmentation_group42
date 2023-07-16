#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/MetricsCalculation.h"

/*
	Given two BB file paths, this function will
	open those files and read each record. It save
	them into two specific structures:

	std::vector<RectangleFileProf> for first input as prof's dataset BB_path
	std::vector<RectangleFileOur> for second input as our's BB_path

	It will return them as a pair, to be studied in second moment
*/
std::pair<std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string profBB_path, std::string ourBB_path)
{
	std::ifstream fileProf(profBB_path);

	std::string lineP;

	std::vector<RectangleFileProf> rectanglesProf;

	// Leggi il primo file e salva i rettangoli
	while (std::getline(fileProf, lineP)) {
		std::stringstream ssP(lineP);
		std::string token;

		// Estrai l'ID
		std::getline(ssP, token, ':'); // Ignora il testo "ID"
		std::getline(ssP, token, ';'); // Estrai il token successivo fino al carattere ';'
		token = token.substr(1); // Ignora lo spazio dopo ':'
		int id = std::stoi(token); // Converte il token in un intero

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


	std::ifstream fileOur(ourBB_path);
	std::string lineO;
	std::vector<RectangleFileOur> rectanglesOur;

	// Leggi il secondo file e salva i rettangoli
	while (std::getline(fileOur, lineO)) {
		std::stringstream ssO(lineO);
		std::string token;

		// Estrai l'ID
		std::getline(ssO, token, ':'); // Ignora il testo "ID"
		std::getline(ssO, token, ';'); // Estrai il token successivo fino al carattere ';'
		token = token.substr(1); // Ignora lo spazio dopo ':'
		int id = std::stoi(token); // Converte il token in un intero

		std::string coords;
		std::getline(ssO, coords, '[');
		std::getline(ssO, coords, ']');

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

	fileProf.close();
	fileOur.close();

	return std::make_pair(rectanglesProf, rectanglesOur);

}



double singlePlateFoodSegmentation_IoUMetric(const std::vector<int>& profBB, const std::vector<int>& ourBB)
{
	int x1 = std::max(profBB[0], ourBB[0]);
	int y1 = std::max(profBB[1], ourBB[1]);
	int x2 = std::min(profBB[0] + profBB[2], ourBB[0] + ourBB[2]);
	int y2 = std::min(profBB[1] + profBB[3], ourBB[1] + ourBB[3]);

	int width = std::max(0, x2 - x1);
	int height = std::max(0, y2 - y1);

	int intersectionPixels = width * height;
	
	int unionPixels = profBB[2] * profBB[3] + ourBB[2] * ourBB[3] - intersectionPixels;

	double iou = (double)intersectionPixels / (double)unionPixels;

	return  iou;

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
	double bMpixels = 0, aMpixels = 0;
	for (int y = 0; y < beforeMask.rows; y++)
		for (int x = 0; x < beforeMask.cols; x++)
			if (beforeMask.at<uchar>(y, x) != 255)
				bMpixels++;

	for (int y = 0; y < afterMask.rows; y++)
		for (int x = 0; x < afterMask.cols; x++)
			if (afterMask.at<uchar>(y, x) != 255)
				aMpixels++;

	double r_i = (aMpixels / bMpixels);
	return r_i;
};

/*
	It calculates the Precision-Recall curve.
	Also it gives us the AP for one single class of objects

	For more info : https: // learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
					https: //towardsdatascience.com/a-better-map-for-object-detection-32662767d424
					https:// jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
*/
double calculateAP(const std::vector<Prediction>& predictions, int classID, int gtNumItemClass_pc)
{
	double ap = 0.0;
	int cumulativeTP = 0;
	int cumulativeFP = 0;

	// Ordina le predizioni in ordine decrescente di confidenza
	std::vector<Prediction> sortedPredictions = predictions;


	// Trova la posizione iniziale degli elementi con classe 'id'
	auto partitionIter = std::partition(sortedPredictions.begin(), sortedPredictions.end(), [classID](const Prediction& pred) {
		return pred.getClassId() == classID;
		});

	// Ordina gli elementi con classe 'id' in ordine decrescente di confidenza
	std::sort(sortedPredictions.begin(), partitionIter, [](const Prediction& a, const Prediction& b) {
		return a.getConfidence() > b.getConfidence();
		});




	// Calcola i valori di precisione e recall per ogni predizione
	std::vector<double> precision;
	std::vector<double> recall;
	std::vector<cv::Point2d> p_r_points; //X recall - Y precision
	for (const Prediction& pred : sortedPredictions) {
		if (pred.getClassId() == classID) {
			if (pred.isTP()) {
				cumulativeTP++;
			}
			else {
				cumulativeFP++;
			}

			double currPrecision = static_cast<double>(cumulativeTP) / (cumulativeTP + cumulativeFP);
			double currRecall = static_cast<double>(cumulativeTP) / gtNumItemClass_pc;

			p_r_points.push_back(cv::Point2d(currRecall, currPrecision));
			//precision.push_back(currPrecision);
			//recall.push_back(currRecall);

		}
	}

	// Calcola l'Average Precision (AP) utilizzando l'interpolazione a 11 punti di recall
	std::vector<double> recallPoints = { 0.0, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000 };
	std::vector<cv::Point2d> precisionRecallCurve;

	// Ordina gli elementi con recall in ordine decrescente di confidenza
	std::sort(p_r_points.begin(), p_r_points.end(), [](const cv::Point2d& a, const cv::Point2d& b) {
		return a.x < b.x;
		});

	for (int i = 0; i < p_r_points.size(); i++)
	{
		int nearestIndex = -1;
		double minDiffSoFar = 2;
		for (int j = 0; j < recallPoints.size(); j++)
		{
			double currDiff = abs(p_r_points.at(i).x - recallPoints.at(j));
			if (currDiff <= minDiffSoFar)
			{
				minDiffSoFar = currDiff;
				nearestIndex = j;
			}
		}
		p_r_points.at(i).x = recallPoints.at(nearestIndex);
	}

	for (int rp = 0; rp < recallPoints.size(); rp++)
	{
		double maxPrecision = -1;

		for (int i = 0; i < p_r_points.size(); i++)
		{
			if (p_r_points.at(i).x == recallPoints.at(rp))
				if (maxPrecision < p_r_points.at(i).y)
					maxPrecision = p_r_points.at(i).y;
		}
		precisionRecallCurve.push_back(cv::Point2d(recallPoints.at(rp),maxPrecision));
	}

	double maxPrecisionSoFar = 0.0;
	for (int rp = precisionRecallCurve.size() - 1; rp >=0; rp--)
	{
		if (precisionRecallCurve.at(rp).y >= maxPrecisionSoFar)
			maxPrecisionSoFar = precisionRecallCurve.at(rp).y;
		else
			precisionRecallCurve.at(rp).y = maxPrecisionSoFar;
	}

	for (int rp = 0; rp < recallPoints.size();)
	{
		double soFarPrec = precisionRecallCurve.at(rp).y;
		int soFarCounter = 0;
		while (rp < recallPoints.size() && precisionRecallCurve.at(rp).y == soFarPrec)
		{
			soFarCounter++;
			rp++;

		}
		
		ap += soFarCounter * soFarPrec;
	}

	return ap/11;
}


/*
	Function that takes in input:

	code: number from 0 (FoodImage) to 3 (Leftover 3)
	gT_Masks: ImageMasks of GroundTruth (prof dataset)
	gT_BBs: path to GroundTruth BBs (prof dataset)
	ourMasks: Our image, with all food masks found
	ourBBs: path to our BBs found

	predictions: a structure to store all food detected by us. Look Prediction class to see more
	predicted: a structure to store all class food we've studied previously

	Optional parameters: (How "code" works).
	We should perform specific analysis on specific results ("Not all to all")
	beforeMasks: ImageMasks of GroundTruth TAKEN AS A "BEFORE IMAGE" (prof dataset)
	beforeBBs: Our image, with all food masks found TAKEN AS A "AFTER IMAGE"
*/
std::pair<double, int> OneImageSegmentation_MetricCalculations_(
	int code,

	//always must-have
	const cv::Mat& gT_FI_masks,
	const std::string gT_FI_BBs,
	const cv::Mat& ourMasks_FI,
	std::string ourBBs_FI,

	std::vector<Prediction>& predictions,
	std::set<int>& predictedClasses,
	std::vector<std::pair<int, int>>& gTfoodItem_numbers,
	int& gtf,

	const cv::Mat& gT_leftover_masks,
	const std::string gT_leftover_BBs,
	const cv::Mat& ourMasks_leftover,
	const std::string ourBBs_leftover
)
{
	std::string output = "\n\nR_i's found: {   ";
	double sum_iou = 0.0;

	int numberFoodItemsSingleImage = 0;

	//IouMetric & AP metric
	if (code == 0)
	{

		std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gT_FI_BBs, ourBBs_FI);
		std::vector<RectangleFileProf> gT_rects_FI = rectsForIoU.first;
		std::vector<RectangleFileOur> our_rects_FI = rectsForIoU.second;

		for (const auto& gT_rect_fi : gT_rects_FI)
		{

		    numberFoodItemsSingleImage++;
			int gTFoodItem_ID = gT_rect_fi.getRectId();
			bool toAdd = true;
			for (auto& pair : gTfoodItem_numbers)
			{
				if (pair.first == gTFoodItem_ID)
				{
					pair.second++;
					toAdd = false;
					break;
				}
			}
			if (toAdd) { gTfoodItem_numbers.push_back(std::make_pair(gTFoodItem_ID, 1)); }

			bool foundMatch = false;

			for (auto& our_rect_fi : our_rects_FI)
			{
				if (gT_rect_fi.getRectId() == our_rect_fi.getRectId())
				{
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					foundMatch = true;
					our_rect_fi.setIsTaken(true);

					double single_iou = singlePlateFoodSegmentation_IoUMetric(gT_rect_fi.getCoords(), our_rect_fi.getCoords());
					our_rect_fi.setPrediction(single_iou);
					sum_iou += single_iou;

					std::cout << "\nFor food item " << our_rect_fi.getRectId() << ", its IoU => " << single_iou << '\n';

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				//std::cout << "\n\n" << "NIENTE MATCH PER CIBO: " << gT_rect_fi.getRectId() << " in " << gT_FI_BBs;
			}

		}

		//AP calculations FI
		for (const auto& r : our_rects_FI)
		{
			int id = r.getRectId();
			predictedClasses.insert(id);
			bool isTruePositive = false;
			if (r.getPrediction() > 0.5) isTruePositive = true;
			predictions.push_back(Prediction(id, r.getPrediction(), isTruePositive));
		}
	}

	else if (code == 1 || code == 2 )
	{
		std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gT_leftover_BBs, ourBBs_leftover);
		std::vector<RectangleFileProf> gT_rects_LO = rectsForIoU.first;
		std::vector<RectangleFileOur> our_rects_LO = rectsForIoU.second;

		for (const auto& gT_rect_lo : gT_rects_LO)
		{
			//"For food localization and food segmentation you need to evaluate your 
			//system on the “before” images and the images for difficulties 1) and 2) "
			if (code == 1 || code == 2)
			{
				numberFoodItemsSingleImage++;
				int gTFoodItem_ID = gT_rect_lo.getRectId();
				bool toAdd = true;
				for (auto& pair : gTfoodItem_numbers)
				{
					if (pair.first == gTFoodItem_ID)
					{
						pair.second++;
						toAdd = false;
						break;
					}
				}
				if (toAdd) { gTfoodItem_numbers.push_back(std::make_pair(gTFoodItem_ID, 1)); }

			}

			bool foundMatch = false;

			for (auto& our_rect_lo : our_rects_LO)
			{
				if (gT_rect_lo.getRectId() == our_rect_lo.getRectId())
				{
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					foundMatch = true;
					our_rect_lo.setIsTaken(true);

					//We evaluate also for 3) because we need the confidence score
					double single_iou = singlePlateFoodSegmentation_IoUMetric(gT_rect_lo.getCoords(), our_rect_lo.getCoords());
					our_rect_lo.setPrediction(single_iou);

					std::cout << "\nFor food item " << our_rect_lo.getRectId() << ", its IoU => " << single_iou << '\n';

					sum_iou += single_iou;

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				//std::cout << "\n" << "NIENTE MATCH PER CIBO: " << gT_rect_lo.getRectId() << " in " << gT_leftover_BBs;
			}

		}

		//AP calculations FI
		for (const auto& r : our_rects_LO)
		{
			int id = r.getRectId();
			predictedClasses.insert(id);
			bool isTruePositive = false;
			if (r.getPrediction() > 0.5) isTruePositive = true;
			predictions.push_back(Prediction(id, r.getPrediction(), isTruePositive));
		}
	}

	//Leftover Estimation
	if (code!=0)
	{
		std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForLE_FI = boundingBoxFileTokenizer(gT_FI_BBs, ourBBs_FI);
		std::pair< std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> rectsForLE_LO = boundingBoxFileTokenizer(gT_leftover_BBs, ourBBs_leftover);
		std::vector<RectangleFileProf> gT_rects_FI = rectsForLE_FI.first;
		std::vector<RectangleFileOur> our_rects_FI = rectsForLE_FI.second;
		std::vector<RectangleFileProf> gT_rects_LO = rectsForLE_LO.first;
		std::vector<RectangleFileOur> our_rects_LO = rectsForLE_LO.second;


		for (const auto& gT_rect_lo : gT_rects_LO)
		{
			bool foundMatch = false;

			for (auto& our_rect_lo : our_rects_LO) 
			{
				if (gT_rect_lo.getRectId() == our_rect_lo.getRectId())
				{
					foundMatch = true;
					our_rect_lo.setIsTaken(true);


					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels

					cv::Mat oneMask_GTLO(gT_leftover_masks.size(), CV_8UC1, cv::Scalar(255));
					cv::Mat oneMask_OURLO(ourMasks_leftover.size(), CV_8UC1, cv::Scalar(255));

					int ourLeftoverPixels = 0;
					int profLeftoverPixels = 0;
					//COMPARIAMO MASCHERE LEFTOVER NOSTRO CON LEFTOVER PROF => R_i CALCULATIONS
					for (int row = our_rect_lo.getCoords().at(1); row < our_rect_lo.getCoords().at(1) + our_rect_lo.getCoords().at(3); row++)
					{
						for (int col = our_rect_lo.getCoords().at(0); col < our_rect_lo.getCoords().at(0) + our_rect_lo.getCoords().at(2); col++)
							if (ourMasks_leftover.at<uchar>(row, col) == (uchar)our_rect_lo.getRectId())
							{
								oneMask_OURLO.at<uchar>(row, col) = (uchar)our_rect_lo.getRectId();
								ourLeftoverPixels++;
							}
					}
					for (int row = gT_rect_lo.getCoords().at(1); row < gT_rect_lo.getCoords().at(1) + gT_rect_lo.getCoords().at(3); row++)
					{
						for (int col = gT_rect_lo.getCoords().at(0); col < gT_rect_lo.getCoords().at(0) + gT_rect_lo.getCoords().at(2); col++)
							if (gT_leftover_masks.at<uchar>(row, col) == (uchar)gT_rect_lo.getRectId())
							{
								oneMask_GTLO.at<uchar>(row, col) = (uchar)gT_rect_lo.getRectId();
								profLeftoverPixels++;
							}
					}

					std::cout << "\nour leftover's pixels for food item " << our_rect_lo.getRectId() << " are: " << ourLeftoverPixels;
					std::cout << "\nABS of the differences in pixels for food item " << our_rect_lo.getRectId() << " is: " << abs(ourLeftoverPixels - profLeftoverPixels);

					for (auto& our_rect_fi : our_rects_FI)
					{
						if (gT_rect_lo.getRectId() == our_rect_fi.getRectId())
						{
							cv::Mat oneMask_OURFI(ourMasks_FI.size(), CV_8UC1, cv::Scalar(255));
							for (int row = our_rect_fi.getCoords().at(1); row < our_rect_fi.getCoords().at(1) + our_rect_fi.getCoords().at(3); row++)
							{
								for (int col = our_rect_fi.getCoords().at(0); col < our_rect_fi.getCoords().at(0) + our_rect_fi.getCoords().at(2); col++)
									if (ourMasks_FI.at<uchar>(row, col) == (uchar)our_rect_fi.getRectId())
									{
										oneMask_OURFI.at<uchar>(row, col) = (uchar)our_rect_fi.getRectId();
									}
							}

							output += std::to_string(singlePlateLeftoverEstimationMetric(oneMask_OURFI, oneMask_OURLO)) + "   ";
						}
					}
					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our image 
				//(BAD NEWS if wrong detection / GOOD NEWS if leftover has no food left)
				output += "0  ";
				//std::cout << "\n" << "NIENTE MATCH PER CIBO: " << gT_rect_lo.getRectId() << " in " << gT_leftover_BBs << "(ORIGINAL IMAGE IS "
					//<< ourBBs_leftover << ")";
			}

		}

		output += "}";
		std::cout << output << "\n\n";
	}

	gtf += numberFoodItemsSingleImage;


	return 	std::make_pair(sum_iou, numberFoodItemsSingleImage);
}