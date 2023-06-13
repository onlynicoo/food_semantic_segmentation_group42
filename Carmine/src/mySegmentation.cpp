#include "mySegmentation.h"
#include "gaborFeatures.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <random>

using namespace cv;
using namespace std;


//Given a number K, it returns a K_Means clustering
//Every cluster/region is colored with its centroid color value
//Return is an RGB image, highlighting each region.
Mat K_Means(const Mat &input_image, int K)
{
	Mat samples(input_image.rows * input_image.cols, input_image.channels(), CV_32F);

	for (int y = 0; y < input_image.rows; y++)
	{
		for (int x = 0; x < input_image.cols; x++)
		{
			for (int z = 0; z < input_image.channels(); z++)
			{
				if (input_image.channels() == 3)
					samples.at<float>(y + x * input_image.rows, z) = input_image.at<Vec3b>(y, x)[z];
				else
					samples.at<float>(y + x * input_image.rows, z) = input_image.at<uchar>(y, x);
			}
		}
	}

	Mat labels, centers;
	int attempts = 5;
	kmeans(samples, K, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


	Mat new_image(input_image.size(), input_image.type());
	for (int y = 0; y < input_image.rows; y++)
		for (int x = 0; x < input_image.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input_image.rows, 0);
			if (input_image.channels() == 3) {
				for (int i = 0; i < input_image.channels(); i++) {
					new_image.at<Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
				}
			}
			else {
				new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
			}
		}
	//imshow("clustered image", new_image);
	return new_image;
}

//Using previous function output, this method extrapolate
//each region separately, and returns all of them in a std::vector
vector<Mat> extractRegionOfKMeans(const Mat& kmimage, const Mat& img)
{
	Mat forNow; cvtColor(kmimage, forNow, COLOR_BGR2GRAY);
	vector<Mat> regions;
	set<uchar> colors;
	for (int y = 0; y < forNow.rows; y++)
	{
		for (int x = 0; x < forNow.cols; x++)
		{
			int previousSize = colors.size();
			uchar uc = forNow.at<uchar>(y, x);
			colors.insert(uc);
			if (colors.size() > previousSize)
			{

				Mat currentColorRegion(kmimage.rows, kmimage.cols, CV_8UC3);
				for (int y_ = 0; y_ < forNow.rows; y_++)
				{
					for (int x_ = 0; x_ < forNow.cols; x_++)
					{
						if (forNow.at<uchar>(y_, x_) == uc)
						{
							currentColorRegion.at<Vec3b>(y_, x_)[0] = img.at<Vec3b>(y_, x_)[0];
							currentColorRegion.at<Vec3b>(y_, x_)[1] = img.at<Vec3b>(y_, x_)[1];
							currentColorRegion.at<Vec3b>(y_, x_)[2] = img.at<Vec3b>(y_, x_)[2];
						}

						else
						{
							currentColorRegion.at<Vec3b>(y_, x_)[0] = 255;
							currentColorRegion.at<Vec3b>(y_, x_)[1] = 255;
							currentColorRegion.at<Vec3b>(y_, x_)[2] = 255;
						}
					}
				}

				regions.push_back(currentColorRegion);
			}

		}
	}
	return regions;
}

//Given p points in dimensional feature space, this method
//return the centroid among all those points
vector<double> calculateCentroid(const vector<vector<double>>& vectors)
{
	int numVectors = vectors.size();
	int vectorSize = vectors[0].size();

	// Inizializza il vettore centroide con zeri
	std::vector<double> centroid(vectorSize, 0.0);

	// Somma i valori corrispondenti degli elementi dei vettori
	for (const auto& vec : vectors)
		for (int i = 0; i < vectorSize; i++)
			centroid[i] += vec[i];

	// Dividi ogni elemento del vettore centroide per il numero totale di vettori
	for (int i = 0; i < vectorSize; i++)
		centroid[i] /= numVectors;

	return centroid;
}

//L2NORM distance of two n-dimensional feature vectors.
//Each feature vector represents a point in n-dim space
double featureColorGaborL2Norm(const vector<double>& fcg1, const vector<double>& fcg2)
{
	double squaredSum = 0.0;

	for (int i = 0; i < fcg1.size(); i++)
		squaredSum += pow(abs(fcg1.at(i) - fcg2.at(i)), 2);

	return sqrt(squaredSum);
}

//WORK IN PROGRESS
vector<Mat> meanShiftGaborColorSegmentation(const vector<Mat>& regions, const vector<vector<double>>& featureSpace)
{
	double r = 407; //window size
	int ck = 1;
	int cn = 1;

	double previous_gradientNorm = 0.0;

	vector<double> current_cursor = calculateCentroid(featureSpace);
	while (true)
	{
		double costant_term = (2 * ck) / (pow(r, 2));
		double k_dens_estimator = 0.0;
		double 	kde_multiplier = 1 / (featureSpace.size() * pow(r, featureSpace[0].size()));

		for (int i = 0; i < featureSpace.size(); i++)
		{
			double y_i = (1 / pow(r, 2)) * featureColorGaborL2Norm(featureSpace[i], current_cursor);
			k_dens_estimator += kde_multiplier * cn * y_i * exp(-0.5 * pow(abs(y_i), 2));
		}

		vector<double> msv;
		vector<double> massCenter;

		double sumOfAllgyis = 0.0;
		for (int j = 0; j < featureSpace.size(); j++)
		{
			double y_i = (1 / pow(r, 2)) * featureColorGaborL2Norm(featureSpace[j], current_cursor);
			double gyi = cn * y_i * exp(-0.5 * pow(abs(y_i), 2));
			sumOfAllgyis += gyi;
		}
		for (int i = 0; i < featureSpace[0].size(); i++)
		{
			double ith_component = 0.0;
			for (int j = 0; j < featureSpace.size(); j++)
			{
				double y_i = (1 / pow(r, 2)) * featureColorGaborL2Norm(featureSpace[j], current_cursor);
				double gyi = cn * y_i * exp(-0.5 * pow(abs(y_i), 2));

				ith_component += gyi * featureSpace[j][i];
			}
			massCenter.push_back(ith_component/sumOfAllgyis);
		}
		for (int i = 0; i < massCenter.size(); i++)
			msv.push_back(massCenter[i] - current_cursor[i]);

		vector<double> gradient;
		for (int i = 0; i < msv.size(); i++)
			gradient.push_back(costant_term * msv[i] * k_dens_estimator);

		for (int i = 0; i < featureSpace[0].size(); i++)
			current_cursor[i] += msv[i]; // Aggiornamento del centroide

		double current_gradientNorm = cv::norm(current_cursor, cv::NORM_L2);

		//If gradient changes too little in this iteration, we've reached convergence
		if ( abs(previous_gradientNorm - current_gradientNorm) <= 0.5 ) break;

		previous_gradientNorm = current_gradientNorm;
	}

	//WE HAVE PEAK COORDINDATE
	double leftHandSide = 0.0;
	vector<bool> toFuse;

	for (int i = 0; i < featureSpace.size(); i++)
	{
		for (int j = 0; j < featureSpace[0].size(); j++)
		{
			leftHandSide += pow(featureSpace[i][j] - current_cursor.at(j), 2);
		}

		cout << sqrt(leftHandSide) << '\n';
		if (sqrt(leftHandSide) <= r)
			toFuse.push_back(true);
		else
			toFuse.push_back(false);
	}

	vector<Mat> to_out; 
	int blue = std::rand() % 255;
	int green = std::rand() % 255;
	int red = std::rand() % 255;
	Mat fusions = Mat(regions[0].rows, regions[0].cols, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < toFuse.size(); i++)
	{


		if (toFuse.at(i) == true)
		{
			Mat current_region = regions.at(i);
			for (int y = 0; y < current_region.rows; y++)
			{
				for (int x = 0; x < current_region.cols; x++)
				{
					Vec3b crv3b = current_region.at<Vec3b>(y, x);
					if (crv3b[0] != 255 && crv3b[1] != 255 && crv3b[2] != 255)
					{
						fusions.at<Vec3b>(y, x)[0] = crv3b[0];
						fusions.at<Vec3b>(y, x)[1] = crv3b[1];
						fusions.at<Vec3b>(y, x)[2] = crv3b[2];
					}
				}
			}

		}
	}

	to_out.push_back(fusions);

	for (int i = 0; i < toFuse.size(); i++)
		if (toFuse.at(i) == false)
			to_out.push_back(regions.at(i));

	return to_out;
}

//WORK IN PROGRESS
void mySeg(const Mat &img) {

	Mat image = img.clone();
	int Clusters = 8;
	Mat Clustered_Image = K_Means(image, Clusters);
	vector<Mat> regions = extractRegionOfKMeans(Clustered_Image,img);

	//Ready for MeanShift

	vector<vector<double>> featureSpace;
	for (const Mat region : regions) { featureSpace.push_back(getColorGaborFeatures(region)); }

	vector<Mat> finalsegmentation = meanShiftGaborColorSegmentation(regions, featureSpace);

	for (int i = 0; i < finalsegmentation.size(); i++)
		imshow(to_string(i), finalsegmentation.at(i));

	waitKey(0);

}