#pragma once
#include <opencv2/opencv.hpp>

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

		//Getters
		int getClassId() const { return classID; }
		double getConfidence() const { return confidence; }
		bool isTP() const { return isTruePositive; }

		//Constructor
		Prediction(int cid, double conf, bool iTP)
			: classID(cid), confidence(conf), isTruePositive(iTP) {}
};

/*
	Structure useful for Metrics Calculation's steps
	It takes into account the
	"ID: id; [x, y, width, heigth]" format

	GT means ground truth, where the BB was taken from

*/
class RectangleFileGT
{
	private:
		int rectID;
		std::vector<int> coordinates;

	public:

		//Getters
		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; }

		//Constructor
		RectangleFileGT(int id, const std::vector<int>& coords)
			: rectID(id), coordinates(coords) {}
};

/*
	Structure useful for for Metrics Calculation's steps
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
		
		//It has been found a match
		bool isTaken;

		//its confidence score
		double prediction;

	public:

		//Getters
		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; }
		bool getIsTaken() const { return isTaken; }
		double getPrediction() const { return prediction; }

		//Setters
		void setIsTaken(bool taken) { isTaken = taken; }
		void setPrediction(double pred) { prediction = pred; }

		//Constructor
		RectangleFileOur(int id, const std::vector<int>& coords, bool tak)
			: rectID(id), coordinates(coords), isTaken(tak), prediction(0) {}
};



/****************************************************************************************************************************/

//Tool to analyze bounding boxes files
std::pair<std::vector<RectangleFileGT>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string, std::string);

//To compute IoU from a single food item
double singlePlateFoodSegmentationIoUMetric(const std::vector<int>&, const std::vector<int>&);

//To compute Ri ratio
double singlePlateLeftoverEstimationMetric(const cv::Mat&, const cv::Mat&);

//platform launch method. Takes in input food images and leftover's BB path and masks
std::pair<double, int> OneImageSegmentationMetricCalculations(
	int,

	const cv::Mat&,
	const std::string,
	const cv::Mat&,
	std::string,

	std::vector<Prediction>&,
	std::set<int>&,
	std::vector<std::pair<int,int>>&,
	int&,

	const cv::Mat&,
	const std::string,
	const cv::Mat&,
	const std::string
);

double calculateAP(const std::vector<Prediction>&, int,int);