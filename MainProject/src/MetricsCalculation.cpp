#include <fstream>
#include <sstream>
#include <vector>
#include <string>
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
			if (singleFoodMasks_prof.at<uchar>(y, x) != 255)
			{
				unionPixels++;
				if (singleFoodMasks_our.at<uchar>(y, x) != 255)
					intersectionPixels++;

			}
			else
				if (singleFoodMasks_our.at<uchar>(y, x) != 255)
					unionPixels++;

	double IoU = (intersectionPixels / unionPixels);
	return IoU;
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
std::pair<double, int> OneImageSegmentation_MetricCalculations(
	int code,
	const cv::Mat& gT_Masks,
	const std::string gT_BBs,
	const cv::Mat& ourMasks,
	std::string ourBBs,
	std::vector<double>& mAPs,
	int& gtf,
	const cv::Mat& beforeMasks, /* gT_FI_masks */
	const std::string beforeBBs /* gT_FI_BB */)
{
	double iou = 0.0;
	int numberFoodsGroundTruth = 0;


	std::vector<Prediction> predictions_oneImage;
	std::set<int> predictedClasses_oneImage;

	std::cout << "Taking: " + gT_BBs + " and " + ourBBs + "\n";
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

			bool foundMatch = false;

			for (auto& our_rect : our_rects)
			{
				if (gT_rect.getRectId() == our_rect.getRectId())
				{
					foundMatch = true;
					our_rect.setIsTaken(true);
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels

					cv::Mat oneMask_Prof(gT_Masks.size(), CV_8UC1, cv::Scalar(255));
					cv::Mat oneMask_Our(ourMasks.size(), CV_8UC1, cv::Scalar(255));

					/*std::cout << ourMasks.type() << gT_Masks.type();
					cv::rectangle(ourMasks, cv::Rect(our_rect.getCoords().at(0), our_rect.getCoords().at(1), our_rect.getCoords().at(2), our_rect.getCoords().at(3)), cv::Scalar(255), 2);
					imshow("iuHIU9", ourMasks);
					cv::waitKey(0);*/
					for (int row = our_rect.getCoords().at(1); row < our_rect.getCoords().at(1) + our_rect.getCoords().at(3); row++)
					{
						for (int col = our_rect.getCoords().at(0); col < our_rect.getCoords().at(0) + our_rect.getCoords().at(2); col++)
							if (ourMasks.at<uchar>(row, col) == (uchar)our_rect.getRectId())
							{
								oneMask_Our.at<uchar>(row, col) = (uchar)our_rect.getRectId();
							}
					}
					for (int row = gT_rect.getCoords().at(1); row < gT_rect.getCoords().at(1) + gT_rect.getCoords().at(3); row++)
					{
						for (int col = gT_rect.getCoords().at(0); col < gT_rect.getCoords().at(0) + gT_rect.getCoords().at(2); col++)
							if (gT_Masks.at<uchar>(row, col) == (uchar)gT_rect.getRectId()) {
								oneMask_Prof.at<uchar>(row, col) = (uchar)gT_rect.getRectId();
							}
					}

					/*
					imshow("our", oneMask_Our);
					imshow("prof", oneMask_Prof);
					cv::waitKey(0);*/


					double single_iou = singlePlateFoodSegmentation_IoUMetric(oneMask_Prof, oneMask_Our);
					if (single_iou == 0)
					{
						imshow("AJAHJA", oneMask_Prof);
						imshow("AJAHJCDA", oneMask_Our);
						cv::waitKey(0);
					}
					our_rect.setPrediction(single_iou);
					iou += single_iou;
					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our image (BAD NEWS)
				// numberFoodsGroundTruth++ && iou+=0 means that our mIoU will be lower
				std::cout << "\n\n" << "NIENTE MATCH PER RECT: " << gT_rect.getRectId() << " in " << gT_BBs;
			}

		}

		for (const auto& r : our_rects)
		{
			int id = r.getRectId();
			predictedClasses_oneImage.insert(id);
			bool isTruePositive = false;
			if (r.getPrediction() > 0.5) isTruePositive = true;
			predictions_oneImage.push_back(Prediction(id, r.getPrediction(), isTruePositive));
		}

		for (const auto& pr : predictedClasses_oneImage)
			mAPs.push_back(calculateAP(predictions_oneImage, pr, predictedClasses_oneImage.size()));

		predictedClasses_oneImage.clear();
		predictions_oneImage.clear();
	}

	//LEFTOVER ESTIMATION PART
	if (code == 1 || code == 2 || code == 3)
	{
		for (const auto& before_rect : before_rects)
		{
			bool foundMatch = false;

			for (auto& after_rect : after_rects) {
				if (before_rect.getRectId() == after_rect.getRectId()) {
					foundMatch = true;
					after_rect.setIsTaken(true);
					//If true there's a match
					//I'll take the two gray levels representing the two masks gray levels

					cv::Mat oneMask_Before(beforeMasks.size(), CV_8UC1, cv::Scalar(255));
					cv::Mat oneMask_After(ourMasks.size(), CV_8UC1, cv::Scalar(255));

					for (int row = after_rect.getCoords().at(1); row < after_rect.getCoords().at(1) + after_rect.getCoords().at(3); row++)
					{
						for (int col = after_rect.getCoords().at(0); col < after_rect.getCoords().at(0) + after_rect.getCoords().at(2); col++)
							if (ourMasks.at<uchar>(row, col) == (uchar)after_rect.getRectId())
							{
								oneMask_After.at<uchar>(row, col) = (uchar)after_rect.getRectId();
							}
					}

					for (int row = before_rect.getCoords().at(1); row < before_rect.getCoords().at(1) + before_rect.getCoords().at(3); row++)
					{
						for (int col = before_rect.getCoords().at(0); col < before_rect.getCoords().at(0) + before_rect.getCoords().at(2); col++)
							if (beforeMasks.at<uchar>(row, col) == (uchar)before_rect.getRectId()) {
								oneMask_Before.at<uchar>(row, col) = (uchar)before_rect.getRectId();
							}
					}




					/*resize(oneMask_After, oneMask_After, cv::Size(oneMask_After.cols / 2, oneMask_After.rows / 2));
					resize(oneMask_Before, oneMask_Before, cv::Size(oneMask_Before.cols / 2, oneMask_Before.rows / 2));
					cv::imshow("IHi9", oneMask_Before);
					cv::imshow("hj9oi0", oneMask_After);
					cv::waitKey(0);*/

					output += std::to_string(singlePlateLeftoverEstimationMetric(oneMask_Before, oneMask_After)) + "   ";
					break;
				}
			}

			if (!foundMatch) {
				//There's no match, no food item founded in our image 
				//(BAD NEWS if wrong detection / GOOD NEWS if leftover has no food left)
				output += "0  ";
				std::cout << "\n\n" << "NIENTE MATCH PER RECT: " << before_rect.getRectId() << " in " << beforeBBs << "(ORIGINAL IMAGE IS "
					<< ourBBs << ")";
			}

		}

		output += "}";
		std::cout << output << "\n\n";
	}
	return 	std::make_pair(iou, numberFoodsGroundTruth);

}



/*
	It calculates the Precision-Recall curve.
	Also it gives us the AP for one single class of objects

	For more info : https: // learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
					https: //towardsdatascience.com/a-better-map-for-object-detection-32662767d424
					https:// jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
*/
double calculateAP(const std::vector<Prediction>& predictions, int classID, int numClassesDetected)
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
			double currRecall = static_cast<double>(cumulativeTP) / predictions.size();

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
		if (abs(p_r_points.at(i).x - floor(p_r_points.at(i).x)) > 0.5)
			p_r_points.at(i).x = floor(p_r_points.at(i).x) + 1;
		else
			p_r_points.at(i).x = floor(p_r_points.at(i).x);


	for (int rp = 0; rp < recallPoints.size(); rp++)
	{
		double maxPrecision = -1;

		for (int i = 0; i < p_r_points.size(); i++)
		{
			if (p_r_points.at(i).x == recallPoints.at(rp))
				if (maxPrecision < p_r_points.at(i).y)
					maxPrecision = p_r_points.at(i).y;
		}
		precisionRecallCurve.push_back(cv::Point2d(recallPoints.at(rp), maxPrecision));
	}

	for (int rp = precisionRecallCurve.size() - 1; rp >= 0; rp--)
	{
		double maxPrecisionSoFar = 0.0;
		if (precisionRecallCurve.at(rp).y <= maxPrecisionSoFar)
			precisionRecallCurve.at(rp).y = maxPrecisionSoFar;
		else
			maxPrecisionSoFar = precisionRecallCurve.at(rp).y;
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

	return ap / 11;
}