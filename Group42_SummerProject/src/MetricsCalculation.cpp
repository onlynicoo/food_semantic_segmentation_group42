// Author: Carmine Graniello

#include "../include/MetricsCalculation.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>


/**
 * Given two BB file paths, this function will open those files
 * and read each record. It save them into two specific structures:
 * std::vector<RectangleFileGT> for first input as Food Leftover Dataset's BB_path
 * std::vector<RectangleFileOur> for second input as our's BB_path
 * It will return them as a pair, to be studied in second moment.
 *
 * @param gtBBpath path to ground truth bounding box .txt file
 * @param ourBBpath path to a bounding box .txt file computed by our system
 * @return a pair of vectors of specific types of rectangle. See RectangleFileGT and RectangleFileOur for more info
 */
std::pair<std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string gtBBpath, std::string ourBBpath)
{
	std::ifstream fileGT(gtBBpath);

	std::string lineGT;

	std::vector<RectangleFileGT> rectanglesGT;

	
	//Read the first file and save its bounding boxes records
	while (std::getline(fileGT, lineGT)) {
		std::stringstream ssGT(lineGT);
		std::string token;

		// Extract the ID
		std::getline(ssGT, token, ':'); // Ignore the text "ID"
		std::getline(ssGT, token, ';'); // Extract the next token, till char ';'
		token = token.substr(1); // Ignore space after char ':'
		int id = std::stoi(token); // Convert the token into an integer value

		std::string coords;
		std::getline(ssGT, coords, '[');
		std::getline(ssGT, coords, ']');

		std::stringstream ssCoords(coords);
		std::string coordToken;

		std::vector<int> coordinates;
		while (std::getline(ssCoords, coordToken, ',')) {
			coordinates.push_back(std::stoi(coordToken));
		}

		if (coordinates.size() == 4) {
			rectanglesGT.push_back(RectangleFileGT(id, coordinates));
		}
	}


	std::ifstream fileOur(ourBBpath);
	std::string lineO;
	std::vector<RectangleFileOur> rectanglesOur;

	//Read the second file and save its bounding boxes records
	while (std::getline(fileOur, lineO)) {
		std::stringstream ssO(lineO);
		std::string token;

		// Extract the ID
		std::getline(ssO, token, ':'); // Ignore the text "ID"
		std::getline(ssO, token, ';'); // Extract the next token, till char ';'
		token = token.substr(1); // Ignore space after char ':'
		int id = std::stoi(token); // Convert the token into an integer value

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

	fileGT.close();
	fileOur.close();

	return std::make_pair(rectanglesGT, rectanglesOur);

}


/**
 * Given two stl vectors of integers, representing a bounding box coordinates record for a food item, we:
 * compute the rectangle, intersection of those two boxes.
 * compute its area, in pixels (intersectionPixels).
 * compute the area of the two boxes' union, in pixels (unionPixels).
 * return the ratio intersectionPixels/unionPixels as IoU OF A SINGLE FOOD ITEM
 * Notice that a single BB coordinates record (e.g: [x, y, w, h]) is composed of:
 * x and y for top left pixel corner coordinates
 * w and h respectively for width and height of the BB itself.
 * 
 * @param gtBB ground truth bounding box coordinates record for a food item
 * @param ourBB bounding box coordinates record for a food item, computed by our system
 * @return a double representing intersection over union for a single food item
 */
double singlePlateFoodSegmentationIoUMetric(const std::vector<int>& gtBB, const std::vector<int>& ourBB)
{
	// Compute intersection rectangle top left corner
	int x1 = std::max(gtBB[0], ourBB[0]);
	int y1 = std::max(gtBB[1], ourBB[1]);
	int x2 = std::min(gtBB[0] + gtBB[2], ourBB[0] + ourBB[2]);
	int y2 = std::min(gtBB[1] + gtBB[3], ourBB[1] + ourBB[3]);

	// Compute intersection rectangle dimensions
	int width = std::max(0, x2 - x1);
	int height = std::max(0, y2 - y1);

	int intersectionPixels = width * height;
	
	int unionPixels = gtBB[2] * gtBB[3] + ourBB[2] * ourBB[3] - intersectionPixels;

	double iou = (double)intersectionPixels / (double)unionPixels;

	return  iou;
}

/**
 * Function that takes in input two masks, respectively the food item in "Food_image" (before image)
 * and the food item in "leftover" (after image). We compare the ratio of the pixels numbers of those masks
 * R_i = #pixels for food i in the "after image" / #pixels for food i in the "before image"
 * 
 * @param beforeMask before food item mask
 * @param afterMask after food item mask
 * @return a double, ratio R_i previously described
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

/**
 * Function that calculates the Precision-Recall curve.
 * It calculates Average Precision (AP) using PASCAL VOC 11 point interpolation method 
 * for one single object class. Useful online resources could be:
 * https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
 * https://towardsdatascience.com/a-better-map-for-object-detection-32662767d424
 * https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
 * We calculate the AP, for each class, referring to all possible matches done during our tests
 * 
 * @param predictions is a vector of Prediction objects. Each one is a match with ground truth (TP or FP), based on a score
 * @param classID the class for which we're calculting the AP (food classes)
 * @param gtNumItemClass_pc the number of occasions of classID objects in ground truth (to calculate Recall)
 * @return a double, AP of the system
 */
double calculateAP(const std::vector<Prediction>& predictions, int classID, int gTNumberItemsPerClass)
{
	double ap = 0.0;
	int cumulativeTP = 0;
	int cumulativeFP = 0;

	// We have to sort the predictions by decreasing order of confidence score
	std::vector<Prediction> sortedPredictions = predictions;

	// We put first element of classID
	auto partitionIter = std::partition(sortedPredictions.begin(), sortedPredictions.end(), [classID](const Prediction& pred) {
		return pred.getClassId() == classID;
		});

	// We sort the predictions by decreasing order of confidence score
	std::sort(sortedPredictions.begin(), partitionIter, [](const Prediction& a, const Prediction& b) {
		return a.getConfidence() > b.getConfidence();
		});


	// Computing precision and recall values for each prediction
	std::vector<double> precision;
	std::vector<double> recall;
	std::vector<cv::Point2d> prPoints; //X recall - Y precision
	for (const Prediction& pred : sortedPredictions) {
		if (pred.getClassId() == classID) {
			if (pred.isTP()) {
				cumulativeTP++;
			}
			else {
				cumulativeFP++;
			}

			double currPrecision = static_cast<double>(cumulativeTP) / (cumulativeTP + cumulativeFP);
			double currRecall = static_cast<double>(cumulativeTP) / gTNumberItemsPerClass;
			prPoints.push_back(cv::Point2d(currRecall, currPrecision));
		}
	}

	// Calculate the Average Precision (AP) using 11-recall points interpolation technique
	std::vector<double> recallPoints = { 0.0, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000 };
	std::vector<cv::Point2d> precisionRecallCurve;

	// Sort each p_r point by decreasing order of confidence
	std::sort(prPoints.begin(), prPoints.end(), [](const cv::Point2d& a, const cv::Point2d& b) {
		return a.x < b.x;
		});

	// Associate each p_r point to the nearest one among the 11 points
	for (int i = 0; i < prPoints.size(); i++)
	{
		int nearestIndex = -1;
		double minDiffSoFar = 2;
		for (int j = 0; j < recallPoints.size(); j++)
		{
			double currDiff = abs(prPoints.at(i).x - recallPoints.at(j));
			if (currDiff <= minDiffSoFar)
			{
				minDiffSoFar = currDiff;
				nearestIndex = j;
			}
		}
		prPoints.at(i).x = recallPoints.at(nearestIndex);
	}

	// For each p_r point we associate as y the highest
	// precision value among all the precision values with same recall
	for (int rp = 0; rp < recallPoints.size(); rp++)
	{
		double maxPrecision = -1;

		for (int i = 0; i < prPoints.size(); i++)
		{
			if (prPoints.at(i).x == recallPoints.at(rp))
				if (maxPrecision < prPoints.at(i).y)
					maxPrecision = prPoints.at(i).y;
		}
		precisionRecallCurve.push_back(cv::Point2d(recallPoints.at(rp),maxPrecision));
	}

	//INTERPOLATION. MAX TO THE RIGHT
	double maxPrecisionSoFar = 0.0;
	for (int rp = precisionRecallCurve.size() - 1; rp >=0; rp--)
	{
		if (precisionRecallCurve.at(rp).y >= maxPrecisionSoFar)
			maxPrecisionSoFar = precisionRecallCurve.at(rp).y;
		else
			precisionRecallCurve.at(rp).y = maxPrecisionSoFar;
	}

	// Sum each contribute from the 11 points
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

	//Normalize by factor 11
	return ap/11;
}

/**
 * During test part, after received a Tray object, and understood from 
 * which tray was taken and which leftover was, we use this function as
 * a launch pad for all performances metrics we've described in our report.
 * For a single Tray objects we launch this function two times:
 * To study and compare all info retrieved from "Food_Image" (Code 0)
 * To study and compare all info retrieved from "leftoverY"	(Code Y)
 * 
 * @param code number from 0 (FoodImage) to 3 (leftover3)
 * @param gTFImasks image mask of a ground truth's food image
 * @param gTFIBBs path to gTFImasks bounding boxes .txt file
 * @param ourMasksFI image mask of food image computed by our system
 * @param ourBBsFI path to ourMasksFI bounding boxes .txt file
 * @param predictions vector in which we save all matches with ground truth food items
 * @param predictedClasses a set of integers representing all classes we've analyzed so far
 * @param gTfoodItemNumbers a vector saving a pair (classID, #occurenciesOfClassID), useful for mAP
 * @param gtf an integer to take count of the Groun Truth Food items we've encountered so far
 * @param gTLeftoverMasks image mask of a ground truth's leftover image
 * @param gTLeftoverBBs path to gT_leftover_masks bounding boxes .txt file
 * @param ourMasksLeftover image mask of leftover image computed by our system
 * @param ourBBsLeftover path to ourMasksLeftover bounding boxes .txt file
 * @return a pair: (sum of all food items' IoU's found in an image, # of food item encountered) => e.g : (2.89477383,4)
 */
std::pair<double, int> OneImageSegmentationMetricCalculations(
	int code,

	const cv::Mat& gTFImasks,
	const std::string gTFIBBs,
	const cv::Mat& ourMasksFI,
	std::string ourBBsFI,

	std::vector<Prediction>& predictions,
	std::set<int>& predictedClasses,
	std::vector<std::pair<int, int>>& gTfoodItemNumbers,
	int& gtf,

	const cv::Mat& gTLeftoverMasks,
	const std::string gTLeftoverBBs,
	const cv::Mat& ourMasksLeftover,
	const std::string ourBBsLeftover
)
{
	std::string output = "\nR_i's found: {   ";
	double sumIoUs = 0.0;

	int numberFoodItemsSingleImage = 0;

	//IouMetric & AP metric
	if (code == 0)
	{

		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gTFIBBs, ourBBsFI);
		std::vector<RectangleFileGT> gTRectsFI = rectsForIoU.first;
		std::vector<RectangleFileOur> ourRectsFI = rectsForIoU.second;

		for (const auto& gTrectfi : gTRectsFI)
		{

		    numberFoodItemsSingleImage++;
			int gTFoodItemID = gTrectfi.getRectId();
			bool toAdd = true;
			for (auto& pair : gTfoodItemNumbers)
			{
				if (pair.first == gTFoodItemID)
				{
					pair.second++;
					toAdd = false;
					break;
				}
			}
			if (toAdd) { gTfoodItemNumbers.push_back(std::make_pair(gTFoodItemID, 1)); }

			bool foundMatch = false;

			for (auto& ouRrectfi : ourRectsFI)
			{
				if (gTrectfi.getRectId() == ouRrectfi.getRectId())
				{
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					foundMatch = true;
					ouRrectfi.setIsTaken(true);

					double single_iou = singlePlateFoodSegmentationIoUMetric(gTrectfi.getCoords(), ouRrectfi.getCoords());
					ouRrectfi.setPrediction(single_iou);
					sumIoUs += single_iou;

					std::cout << "\nFood " << ouRrectfi.getRectId() << ", IoU = " << single_iou;

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
			}

		}

		//AP calculations FI
		for (const auto& r : ourRectsFI)
		{
			int id = r.getRectId();
			predictedClasses.insert(id);
			bool isTruePositive = false;
			if (r.getPrediction() > 0.5) isTruePositive = true;
			predictions.push_back(Prediction(id, r.getPrediction(), isTruePositive));
		}
	}

	else if (code == 1 || code == 2)
	{
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gTLeftoverBBs, ourBBsLeftover);
		std::vector<RectangleFileGT> gTRectsLO = rectsForIoU.first;
		std::vector<RectangleFileOur> ourRectsLO = rectsForIoU.second;

		for (const auto& gTrectlo : gTRectsLO)
		{
			//"For food localization and food segmentation you need to evaluate your 
			//system on the �before� images and the images for difficulties 1) and 2) "
			if (code == 1 || code == 2)
			{
				numberFoodItemsSingleImage++;
				int gTFoodItemID = gTrectlo.getRectId();
				bool toAdd = true;
				for (auto& pair : gTfoodItemNumbers)
				{
					if (pair.first == gTFoodItemID)
					{
						pair.second++;
						toAdd = false;
						break;
					}
				}
				if (toAdd) { gTfoodItemNumbers.push_back(std::make_pair(gTFoodItemID, 1)); }

			}

			bool foundMatch = false;

			for (auto& ouRrectlo : ourRectsLO)
			{
				if (gTrectlo.getRectId() == ouRrectlo.getRectId())
				{
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels
					foundMatch = true;
					ouRrectlo.setIsTaken(true);

					//We evaluate also for 3) because we need the confidence score
					double singleIoU = singlePlateFoodSegmentationIoUMetric(gTrectlo.getCoords(), ouRrectlo.getCoords());
					ouRrectlo.setPrediction(singleIoU);

					std::cout << "\nFood " << ouRrectlo.getRectId() << ", IoU = " << singleIoU;

					sumIoUs += singleIoU;

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
			}

		}

		//AP calculations FI
		for (const auto& r : ourRectsLO)
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
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForLEFI = boundingBoxFileTokenizer(gTFIBBs, ourBBsFI);
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForLELO = boundingBoxFileTokenizer(gTLeftoverBBs, ourBBsLeftover);
		std::vector<RectangleFileGT> gTRectsFI = rectsForLEFI.first;
		std::vector<RectangleFileOur> ourRectsFI = rectsForLEFI.second;
		std::vector<RectangleFileGT> gTRectsLO = rectsForLELO.first;
		std::vector<RectangleFileOur> ourRectsLO = rectsForLELO.second;


		for (const auto& gTrectlo : gTRectsLO)
		{
			double groundTruthRI = 0;
			double ourRI = 0;

			for (const auto& gTrectsfi : gTRectsFI)
			{
				if (gTrectlo.getRectId() == gTrectsfi.getRectId())
				{
					cv::Mat oneMaskGTLO(gTLeftoverMasks.size(), CV_8UC1, cv::Scalar(255));
					cv::Mat oneMaskGTFI(gTFImasks.size(), CV_8UC1, cv::Scalar(255));

					//TAKING GROUND TRUTH LEFTOVER MASK
					for (int row = gTrectlo.getCoords().at(1); row < gTrectlo.getCoords().at(1) + gTrectlo.getCoords().at(3); row++)
					{
						for (int col = gTrectlo.getCoords().at(0); col < gTrectlo.getCoords().at(0) + gTrectlo.getCoords().at(2); col++)
							if (gTLeftoverMasks.at<uchar>(row, col) == (uchar)gTrectlo.getRectId())
							{
								oneMaskGTLO.at<uchar>(row, col) = (uchar)gTrectlo.getRectId();
							}
					}
					
					//TAKING GROUND TRUTH LEFTOVER MASK
					for (int row = gTrectsfi.getCoords().at(1); row < gTrectsfi.getCoords().at(1) + gTrectsfi.getCoords().at(3); row++)
					{
						for (int col = gTrectsfi.getCoords().at(0); col < gTrectsfi.getCoords().at(0) + gTrectsfi.getCoords().at(2); col++)
							if (gTFImasks.at<uchar>(row, col) == (uchar)gTrectsfi.getRectId())
							{
								oneMaskGTFI.at<uchar>(row, col) = (uchar)gTrectsfi.getRectId();
							}
					}

					//COMPARING THEM. gtR_I CALCULATION
					groundTruthRI = singlePlateLeftoverEstimationMetric(oneMaskGTFI, oneMaskGTLO);
				}
			}

			for (auto& ourrectlo : ourRectsLO)
			{
				if (gTrectlo.getRectId() == ourrectlo.getRectId())
				{
					for (auto& ourrectfi : ourRectsFI)
					{
						if (ourrectfi.getRectId() == ourrectlo.getRectId())
						{
							cv::Mat oneMaskOURLO(ourMasksLeftover.size(), CV_8UC1, cv::Scalar(255));
							cv::Mat oneMaskOURFI(ourMasksFI.size(), CV_8UC1, cv::Scalar(255));


							for (int row = ourrectlo.getCoords().at(1); row < ourrectlo.getCoords().at(1) + ourrectlo.getCoords().at(3); row++)
							{
								for (int col = ourrectlo.getCoords().at(0); col < ourrectlo.getCoords().at(0) + ourrectlo.getCoords().at(2); col++)
									if (ourMasksLeftover.at<uchar>(row, col) == (uchar)ourrectlo.getRectId())
									{
										oneMaskOURLO.at<uchar>(row, col) = (uchar)ourrectlo.getRectId();
									}
							}
							for (int row = ourrectfi.getCoords().at(1); row < ourrectfi.getCoords().at(1) + ourrectfi.getCoords().at(3); row++)
							{
								for (int col = ourrectfi.getCoords().at(0); col < ourrectfi.getCoords().at(0) + ourrectfi.getCoords().at(2); col++)
									if (ourMasksFI.at<uchar>(row, col) == (uchar)ourrectfi.getRectId())
									{
										oneMaskOURFI.at<uchar>(row, col) = (uchar)ourrectfi.getRectId();
									}
							}

							ourRI = singlePlateLeftoverEstimationMetric(oneMaskOURFI, oneMaskOURLO);

							std::cout << "\n\nFood item " << gTrectlo.getRectId() << " : " << "our R_i = " << ourRI << "\n";
							std::cout << "Food item " << gTrectlo.getRectId() << " : " << "gT R_i = " << groundTruthRI << "\n";
							std::cout << "Food item " << gTrectlo.getRectId() << " : " << "abs diff = " << abs(ourRI - groundTruthRI)<<"\n";
							break;
						}
					}
				}

			}
		}
	}

	gtf += numberFoodItemsSingleImage;

	return 	std::make_pair(sumIoUs, numberFoodItemsSingleImage);
}