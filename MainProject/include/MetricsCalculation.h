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

		Prediction(int cid, double conf, bool iTP)
			: classID(cid), confidence(conf), isTruePositive(iTP) {}
};


/*
	Structure useful for Metrics Calculation's steps
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

		//Getters
		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; }

		RectangleFileProf(int id, const std::vector<int>& coords)
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
		bool isTaken;
		double prediction;

	public:

		//Getters
		int getRectId() const { return rectID; }
		std::vector<int> getCoords() const { return coordinates; }
		bool getIsTaken() const { return isTaken; }
		double getPrediction() const { return prediction; }

		//Setters
		void setIsTaken(bool taken) { isTaken = taken; }
		void setPrediction(bool pred) { prediction = pred; }

		RectangleFileOur(int id, const std::vector<int>& coords, bool tak)
			: rectID(id), coordinates(coords), isTaken(tak), prediction(0) {}
};



/****************************************************************************************************************************/


std::pair<std::vector<RectangleFileProf>, std::vector<RectangleFileOur>> boundingBoxFileTokenizer(std::string, std::string);

void calculateRectangleCentralPixel(int, int, int, int, int&, int&);




double singlePlateFoodSegmentation_IoUMetric(const cv::Mat&, const cv::Mat&);

double singlePlateLeftoverEstimationMetric(const cv::Mat&, const cv::Mat&);



//SOON TO BE DELETED
void OneImageSegmentation_LeftoverEstimation(const cv::Mat&, const cv::Mat&, std::string, std::string);
//SOON TO BE DELETED
std::pair<double, int> OneImageSegmentation_IoUMetric(const cv::Mat&, const cv::Mat&, std::string, std::string, std::vector<Prediction>&, std::set<int>&);


std::pair<double, int> OneImageSegmentation_MetricCalculations(
	int,
	const cv::Mat&,
	const std::string,
	const cv::Mat&,
	std::string,
	std::vector<Prediction>&,
	std::set<int>&,
	const cv::Mat & = cv::Mat(), /* gT_FI_masks */
	const std::string = std::string() /* gT_FI_BB */
);


double calculateAP(const std::vector<Prediction>&, int);