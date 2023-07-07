#include "../include/Test.h"
#include "../include/Tray.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

//to have mIoU we have to do with all masks of a specific class food
double singleFoodSegmentation_IoUMetric(const cv::Mat& groundTruthMask, const cv::Mat& ourMask)
{
	if (groundTruthMask.size() != ourMask.size())
		return -1;

	double intersectionPixels = 0;
	double unionPixels = 0;

	for (int y = 0; y < groundTruthMask.rows; y++)
		for (int x = 0; x < groundTruthMask.cols; x++)
			if (groundTruthMask.at<uchar>(y, x) != 0)
			{
				unionPixels++;
				if (ourMask.at<uchar>(y, x) != 0)
					intersectionPixels++;

			}
			else
				if (ourMask.at<uchar>(y, x) != 0)
					unionPixels++;

	double IoU = (double)(intersectionPixels / unionPixels);
	return IoU;
}


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


Test::Test(std::vector<Tray> vecTray) :trayVector{ vecTray } {}

void Test::test_the_system(const std::string& dataSetPath)
{
	std::vector<std::string> trayNames = { "food_image", "leftover1", "leftover2", "leftover3" };

	std::vector<std::string> boundingBoxesPath;
	std::vector<std::string> masksPath;

	for (int i = 1; i <= 8; i++)
	{
		std::string bb_prof_path = dataSetPath + "/tray" + std::to_string(i + 1) + "/bounding_boxes/";
		std::string mask_prof_path = dataSetPath + "/tray" + std::to_string(i + 1) + "/masks/";
		for (int j = 0; j < trayNames.size(); j++)
		{
			bb_prof_path += trayNames[j] + "_bounding_box.txt";

			if (j == 0)
				mask_prof_path += "food_image_mask.png";
			else
				mask_prof_path += trayNames[j] + ".png";

			boundingBoxesPath.push_back(bb_prof_path);
			masksPath.push_back(mask_prof_path);
		}

	}


	//We have our trayVector {(FOOD1/LEFTOVER1_F1),(FOOD1/LEFTOVER2_F1),(FOOD1/LEFTOVER3_F1),(FOOD2/LEFTOVER1_F2),...}
	for(const auto& tray: trayVector)
	{
		;
	}

}
