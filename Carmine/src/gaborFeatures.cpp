#include "gaborFeatures.h"
#include <opencv2/opencv.hpp>

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
	to_out_features.push_back(colorRegion[0]); //B
	to_out_features.push_back(colorRegion[1]); //G
	to_out_features.push_back(colorRegion[2]); //R
	to_out_features.push_back(computeEnergy(normalizedImage));
	to_out_features.push_back(computeVariance(normalizedImage));
	to_out_features.push_back(computeHomogeneity(normalizedImage));
	to_out_features.push_back(computeContrast(normalizedImage));
	to_out_features.push_back(computeDominantFrequency(normalizedImage));
	to_out_features.push_back(computeDominantOrientation(normalizedImage));

	return to_out_features;
}