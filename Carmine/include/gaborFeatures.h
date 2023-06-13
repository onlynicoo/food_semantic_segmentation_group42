//#ifndef GABORFEATURES_H // Header guard to prevent multiple inclusions
//#define GABORFEATURES_H

#include <opencv2/opencv.hpp>

// Funzione per calcolare l'energia delle risposte del filtro di Gabor
double computeEnergy(const cv::Mat&);

// Funzione per calcolare la varianza delle risposte del filtro di Gabor
double computeVariance(const cv::Mat&);

// Funzione per calcolare l'omogeneità delle risposte del filtro di Gabor
double computeHomogeneity(const cv::Mat&);

double computeDominantFrequency(const cv::Mat&);

double computeDominantOrientation(const cv::Mat&);

// Funzione per calcolare il contrasto delle risposte del filtro di Gabor
double computeContrast(const cv::Mat&);

std::vector<double> getColorGaborFeatures(const cv::Mat& region);

std::vector<double> calculateColorFeatures(const cv::Mat& image);

//#endif // End of header guard