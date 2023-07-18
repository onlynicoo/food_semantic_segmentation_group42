#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "../include/MetricsCalculation.h"

/**
 * Given two BB file paths, this function will open those files
 * and read each record. It save them into two specific structures:
 * std::vector<RectangleFileGT> for first input as Food Leftover Dataset's BB_path
 * std::vector<RectangleFileOur> for second input as our's BB_path
 * It will return them as a pair, to be studied in second moment.
 *
 * @param gtBB_path path to ground truth bounding box .txt file
 * @param ourBB_path path to a bounding box .txt file computed by our system
 * @return a pair of vectors of specific types of rectangle. See RectangleFileGT and RectangleFileOur for more info
 */
std::pair<std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string gtBB_path, std::string ourBB_path)
{
	std::ifstream fileGT(gtBB_path);

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


	std::ifstream fileOur(ourBB_path);
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
double singlePlateFoodSegmentation_IoUMetric(const std::vector<int>& gtBB, const std::vector<int>& ourBB)
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
double calculateAP(const std::vector<Prediction>& predictions, int classID, int gtNumItemClass_pc)
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
		}
	}

	// Calculate the Average Precision (AP) using 11-recall points interpolation technique
	std::vector<double> recallPoints = { 0.0, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000 };
	std::vector<cv::Point2d> precisionRecallCurve;

	// Sort each p_r point by decreasing order of confidence
	std::sort(p_r_points.begin(), p_r_points.end(), [](const cv::Point2d& a, const cv::Point2d& b) {
		return a.x < b.x;
		});

	// Associate each p_r point to the nearest one among the 11 points
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

	// For each p_r point we associate as y the highest
	// precision value among all the precision values with same recall
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
 * @param gT_FI_masks image mask of a ground truth's food image
 * @param gT_FI_BBs path to gT_FI_masks bounding boxes .txt file
 * @param ourMasks_FI image mask of food image computed by our system
 * @param ourBBs_FI path to ourMasks_FI bounding boxes .txt file
 * @param predictions vector in which we save all matches with ground truth food items
 * @param predictedClasses a set of integers representing all classes we've analyzed so far
 * @param gTfoodItem_numbers a vector saving a pair (classID, #occurenciesOfClassID), useful for mAP
 * @param gtf an integer to take count of the Groun Truth Food items we've encountered so far
 * @param gT_leftover_masks image mask of a ground truth's leftover image
 * @param gT_leftover_BBs path to gT_leftover_masks bounding boxes .txt file
 * @param ourMasks_leftover image mask of leftover image computed by our system
 * @param ourBBs_leftover path to ourMasks_leftover bounding boxes .txt file
 * @return a pair: (sum of all food items' IoU's found in an image, # of food item encountered) => e.g : (2.89477383,4)
 */
std::pair<double, int> OneImageSegmentation_MetricCalculations_(
	int code,

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
	std::string output = "\nR_i's found: {   ";
	double sum_iou = 0.0;

	int numberFoodItemsSingleImage = 0;

	//IouMetric & AP metric
	if (code == 0)
	{

		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gT_FI_BBs, ourBBs_FI);
		std::vector<RectangleFileGT> gT_rects_FI = rectsForIoU.first;
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

					std::cout << "\nFood " << our_rect_fi.getRectId() << ", IoU = " << single_iou;

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				//std::cout << "\n\n" << "NO MATCH FOR FOOD: " << gT_rect_fi.getRectId() << " in " << gT_FI_BBs;
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
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForIoU = boundingBoxFileTokenizer(gT_leftover_BBs, ourBBs_leftover);
		std::vector<RectangleFileGT> gT_rects_LO = rectsForIoU.first;
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

					std::cout << "\nFood " << our_rect_lo.getRectId() << ", IoU = " << single_iou;

					sum_iou += single_iou;

					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our FOOD IMAGE (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				//std::cout << "\n" << "NO MATCH FOR FOOD: " << gT_rect_lo.getRectId() << " in " << gT_leftover_BBs;
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
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForLE_FI = boundingBoxFileTokenizer(gT_FI_BBs, ourBBs_FI);
		std::pair< std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> rectsForLE_LO = boundingBoxFileTokenizer(gT_leftover_BBs, ourBBs_leftover);
		std::vector<RectangleFileGT> gT_rects_FI = rectsForLE_FI.first;
		std::vector<RectangleFileOur> our_rects_FI = rectsForLE_FI.second;
		std::vector<RectangleFileGT> gT_rects_LO = rectsForLE_LO.first;
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
					int gTLeftoverPixels = 0; 
					// COMPARING OUR LEFTOVER MASKS WITH GROUND TRUTH'S ONES => R_i CALCULATIONS
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
								gTLeftoverPixels++;
							}
					}

					std::cout << "\nLeftover's pixels. Food " << our_rect_lo.getRectId() << ": " << ourLeftoverPixels;
					std::cout << "\nABS of difference. Food " << our_rect_lo.getRectId() << ": " << abs(ourLeftoverPixels - gTLeftoverPixels);

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
				//std::cout << "\n" << "NO MATCH FOR FOOD: " << gT_rect_lo.getRectId() << " in " << gT_leftover_BBs << "(ORIGINAL IMAGE IS "
					//<< ourBBs_leftover << ")";
			}

		}

		output += "}";
		std::cout << output << "\n";
	}

	gtf += numberFoodItemsSingleImage;


	return 	std::make_pair(sum_iou, numberFoodItemsSingleImage);
}