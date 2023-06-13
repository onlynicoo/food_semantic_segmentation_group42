#include "gaborFeatures.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Funzione per calcolare l'energia delle risposte del filtro di Gabor
double computeEnergy(const cv::Mat& gaborResponses) {
	cv::Mat squaredResponses;
	multiply(gaborResponses, gaborResponses, squaredResponses);
	double meanEnergy = mean(squaredResponses)[0];
	return meanEnergy;
}

// Funzione per calcolare la varianza delle risposte del filtro di Gabor
double computeVariance(const cv::Mat& gaborResponses) {
	cv::Scalar meanValue, stdDev;
	meanStdDev(gaborResponses, meanValue, stdDev);
	double variance = stdDev.val[0] * stdDev.val[0];
	return variance;
}

// Funzione per calcolare l'omogeneità delle risposte del filtro di Gabor
double computeHomogeneity(const cv::Mat& gaborResponses) {
	cv::Mat normalizedResponses;
	normalize(gaborResponses, normalizedResponses, 0, 255, cv::NORM_MINMAX);
	cv::Mat hist;
	int channels[] = { 0 };
	int histSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::calcHist(&normalizedResponses, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
	hist /= normalizedResponses.total();
	double homogeneity = sum(hist.mul(hist))[0];
	return homogeneity;
}

double computeDominantFrequency(const cv::Mat& gaborResponses) {
	// Verifica se l'immagine di input è vuota o nulla
	if (gaborResponses.empty()) {
		// Gestione dell'errore: immagine di input vuota o nulla
		// Restituisci un valore di default o solleva un'eccezione
		return 0.0; // Modifica di default, puoi sostituire con il valore corretto o un altro comportamento desiderato
	}

	// Calcola l'energia media delle risposte del filtro di Gabor
	double totalEnergy = 0.0;
	for (int y = 0; y < gaborResponses.rows; y++) {
		for (int x = 0; x < gaborResponses.cols; x++) {
			totalEnergy += pow(gaborResponses.at<uchar>(y, x), 2);
		}
	}
	double meanEnergy = totalEnergy / (static_cast<double>(gaborResponses.rows) * static_cast<double>(gaborResponses.cols));

	// Calcola la frequenza dominante in base all'energia media
	double dominantFrequency = sqrt(meanEnergy);
	return dominantFrequency;
}

double computeDominantOrientation(const cv::Mat& gaborResponses) {
	// Verifica se l'immagine di input è vuota o nulla
	if (gaborResponses.empty()) {
		// Gestione dell'errore: immagine di input vuota o nulla
		// Restituisci un valore di default o solleva un'eccezione
		return 0.0; // Modifica di default, puoi sostituire con il valore corretto o un altro comportamento desiderato
	}

	// Trova la posizione del valore massimo nella matrice di risposte del filtro di Gabor
	double maxVal;
	cv::Point maxLoc;
	cv::minMaxLoc(gaborResponses, nullptr, &maxVal, nullptr, &maxLoc);

	// Calcola l'orientazione dominante in base alla posizione del valore massimo
	double dominantOrientation = maxLoc.y * 180.0 / gaborResponses.rows;
	return dominantOrientation;
}

// Funzione per calcolare il contrasto delle risposte del filtro di Gabor
double computeContrast(const cv::Mat& gaborResponses) {
	double maxResponse, minResponse;
	cv::minMaxLoc(gaborResponses, &minResponse, &maxResponse);
	double contrast = maxResponse - minResponse;
	return contrast;
}

//Funzione per calcolare
std::vector<double> getColorGaborFeatures(const cv::Mat& region)
{
	cv::Mat imgGray;
	cvtColor(region, imgGray, cv::COLOR_BGR2GRAY);

	cv::Vec3b colorRegion;

	// GET COLOR FEATURES
	bool taken = false;
	for (int y = 0; y < region.rows; y++)
	{
		if (taken) break;

		for (int x = 0; x < region.cols; x++)
		{
			if (taken) break;
			if (imgGray.at<uchar>(y, x) != 0)
			{
				colorRegion[0] = region.at<cv::Vec3b>(y, x)[0];
				colorRegion[1] = region.at<cv::Vec3b>(y, x)[1];
				colorRegion[2] = region.at<cv::Vec3b>(y, x)[2];
				taken = true;
			}

		}
	}

	double frequency = 0.6;
	double theta = 0.8;
	double sigma = 2.0;
	double lambda = 4.0;
	double gamma = 0.5;
	int ksize = static_cast<int>(3 * sigma);

	cv::Mat kernel = cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambda, gamma);

	cv::Mat gaborFilteredImage;
	cv::filter2D(imgGray, gaborFilteredImage, CV_32F, kernel);

	cv::Mat normalizedImage;
	cv::normalize(gaborFilteredImage, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

	std::vector<double> to_out_features;

	// Riempimento del vettore to_out_features con le caratteristiche del colore
	/*to_out_features.push_back(colorRegion[0]); //B
	to_out_features.push_back(colorRegion[1]); //G
	to_out_features.push_back(colorRegion[2]); //R*/
	//to_out_features.push_back(computeEnergy(normalizedImage));
	to_out_features.push_back(computeVariance(normalizedImage));
	//to_out_features.push_back(computeHomogeneity(normalizedImage)*8.5);
	//to_out_features.push_back(computeContrast(normalizedImage));
	to_out_features.push_back(computeDominantFrequency(normalizedImage));
	to_out_features.push_back(computeDominantOrientation(normalizedImage));
	vector<double> colorfeatures = calculateColorFeatures(region);
	for (const double c : colorfeatures) { to_out_features.push_back(c); }

	return to_out_features;
}


vector<double> extractGaborFeatures(const Mat& image, const vector<Mat>& gaborFilters)
{
	vector<double> features;

	for (const auto& filter : gaborFilters) {
		Mat filteredImage;
		filter2D(image, filteredImage, CV_32F, filter);

		Scalar meanValue, stdDev;
		meanStdDev(filteredImage, meanValue, stdDev);

		features.push_back(meanValue[0]);
		features.push_back(stdDev[0]);
	}

	return features;
}

double calculateMeanRGB(const Mat& image) {
	double sumR = 0.0, sumG = 0.0, sumB = 0.0;
	int totalPixels = image.rows * image.cols;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			Vec3b pixel = image.at<Vec3b>(y, x);
			sumR += pixel[2];  // Red channel
			sumG += pixel[1];  // Green channel
			sumB += pixel[0];  // Blue channel
		}
	}

	double meanR = sumR / totalPixels;
	double meanG = sumG / totalPixels;
	double meanB = sumB / totalPixels;

	return (meanR + meanG + meanB) / 3.0;
}

// Calcola la deviazione standard dei canali RGB
double calculateStdDevRGB(const Mat& image) {
	Scalar mean, stddev;
	meanStdDev(image, mean, stddev);
	double stdDevR = stddev[0];
	double stdDevG = stddev[1];
	double stdDevB = stddev[2];
	return (stdDevR + stdDevG + stdDevB) / 3.0;
}

// Calcola l'istogramma normalizzato di un canale di colore
Mat calculateNormalizedHistogram(const Mat& image, int channel) {
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	Mat hist;
	calcHist(&image, 1, &channel, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat normalizedHist;
	normalize(hist, normalizedHist, 0, 1, NORM_MINMAX);

	return normalizedHist;
}

// Calcola la media dei picchi degli istogrammi dei canali RGB
double calculateMeanPeak(const Mat& image) {
	Mat histR = calculateNormalizedHistogram(image, 0);
	Mat histG = calculateNormalizedHistogram(image, 1);
	Mat histB = calculateNormalizedHistogram(image, 2);

	double meanPeakR = mean(histR)[0];
	double meanPeakG = mean(histG)[0];
	double meanPeakB = mean(histB)[0];

	return (meanPeakR + meanPeakG + meanPeakB) / 3.0;
}

double calculateColorProportion(const Mat& image, const Scalar& colorLow, const Scalar& colorHigh) {
	int totalPixels = image.rows * image.cols;
	int count = 0;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			Vec3b pixel = image.at<Vec3b>(y, x);
			if (pixel[0] >= colorLow[0] && pixel[0] <= colorHigh[0] &&
				pixel[1] >= colorLow[1] && pixel[1] <= colorHigh[1] &&
				pixel[2] >= colorLow[2] && pixel[2] <= colorHigh[2]) {
				count++;
			}
		}
	}

	return static_cast<double>(count) / totalPixels;
}


// Funzione principale per calcolare tutte le color features
std::vector<double> calculateColorFeatures(const Mat& image) {
	std::vector<double> colorFeatures;

	double meanRGB = calculateMeanRGB(image);
	colorFeatures.push_back(meanRGB);

	double stdDevRGB = calculateStdDevRGB(image);
	colorFeatures.push_back(stdDevRGB*10);

	double meanPeak = calculateMeanPeak(image);
	colorFeatures.push_back(meanPeak*1000);

	Scalar whiteLow = Scalar(220, 220, 220);
	Scalar whiteHigh = Scalar(255, 255, 255);
	double whiteProportion = calculateColorProportion(image, whiteLow, whiteHigh);
	colorFeatures.push_back(whiteProportion*100);

	Scalar grayLow = Scalar(50, 50, 50);
	Scalar grayHigh = Scalar(200, 200, 200);
	double grayProportion = calculateColorProportion(image, grayLow, grayHigh);
	colorFeatures.push_back(grayProportion*10000);

	Scalar brownLower = Scalar(109, 49, 9);
	Scalar brownUpper = Scalar(169, 89, 29);
	double brownProportion = calculateColorProportion(image, brownLower, brownUpper);
	colorFeatures.push_back(brownProportion);

	Scalar yellowLower = Scalar(200, 200, 0);
	Scalar yellowUpper = Scalar(255, 255, 100);
	double yellowProportion = calculateColorProportion(image, yellowLower, yellowUpper);
	colorFeatures.push_back(yellowProportion);

	Scalar redLower = Scalar(200, 0, 0);
	Scalar redUpper = Scalar(255, 100, 100);
	double redProportion = calculateColorProportion(image, redLower, redUpper);
	colorFeatures.push_back(redProportion);

	Scalar greenLower = Scalar(62, 121, 62);
	Scalar greenUpper = Scalar(100, 255, 100);
	double greenProportion = calculateColorProportion(image, greenLower, greenUpper);
	colorFeatures.push_back(greenProportion*10000);

	return colorFeatures;
}
