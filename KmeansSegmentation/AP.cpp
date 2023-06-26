#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <random>

using namespace cv;
using namespace std;


void mySeg(Mat& input_image, int xc, int yc, int rc)
{
	// Dividi l'immagine RGBA nei canali di colore separati
	std::vector<cv::Mat> channels(4);
	cv::split(input_image, channels);

	// Applica il filtro bilaterale a ciascun canale di colore separatamente
	int diameter = 9;   // Diametro del filtro
	double sigma_color = 30;   // Deviazione standard del dominio
	double sigma_space = 50;   // Deviazione standard dello spazio

	for (int i = 0; i < 3; i++) {
		cv::Mat filtered_channel;
		cv::bilateralFilter(channels[i], filtered_channel, diameter, sigma_color, sigma_space);
		filtered_channel.copyTo(channels[i]);
	}

	// Ricostruisci l'immagine RGBA dai canali di colore filtrati
	cv::Mat filtered_image;
	cv::merge(channels, filtered_image);

	imshow("clustered image", filtered_image);
	waitKey(0);

	int nonTraspCounter = 0;
	for (int y = 0; y < input_image.rows ; y++)
		for (int x = 0; x < input_image.cols; x++)
			if (filtered_image.at<Vec4b>(y, x)[3] == 255) nonTraspCounter++;


	int K = 2;
	Mat samples(nonTraspCounter, 81 , CV_32F);
	int sampleIndex = 0;

	Mat filteredImageHSV;
	cvtColor(filtered_image, filteredImageHSV, COLOR_BGR2HSV);
	imshow("hsv", filteredImageHSV);
	waitKey(0);

	for (int y = 1; y < input_image.rows-1; y++)
	{
		for (int x = 1; x < input_image.cols-1; x++)
		{
			int counter = 0;
			for (int y_ker = y - 1; y_ker <= y + 1; y_ker++)
			{
				for (int x_ker = x - 1; x_ker <= x + 1; x_ker++)
				{
					if (pow(xc - x, 2) + pow(yc - y, 2) <= pow(rc, 2))
					{
						// Calcola la deviazione standard dei canali di colore
						Scalar mean, stddev;
						meanStdDev(filteredImageHSV(Rect(x - 1, y - 1, 3, 3)), mean, stddev);

						// Calcola la varianza dei canali di colore
						float varianceB = stddev.val[0] * stddev.val[0] / 10; // Varianza del canale Blu
						float varianceG = stddev.val[1] * stddev.val[1]/10; // Varianza del canale Verde
						float varianceR = stddev.val[2] * stddev.val[2]/10; // Varianza del canale Rosso*/

						// Aggiungi le caratteristiche di colore e posizione alla matrice samples
						samples.at<float>(sampleIndex, counter++) = filtered_image.at<Vec4b>(y_ker, x_ker)[0]; // Canale Blu
						samples.at<float>(sampleIndex, counter++) = filtered_image.at<Vec4b>(y_ker, x_ker)[1];	// Canale Verde
						samples.at<float>(sampleIndex, counter++) = filtered_image.at<Vec4b>(y_ker, x_ker)[2];	// Canale Rosso
						samples.at<float>(sampleIndex, counter++) = sqrt(pow(xc - x, 2) + pow(yc - y, 2));

						// Salva le varianze nel vettore samples
						samples.at<float>(sampleIndex, counter++) = varianceB;
						samples.at<float>(sampleIndex, counter++) = varianceG;
						samples.at<float>(sampleIndex, counter++) = varianceR;
						samples.at<float>(sampleIndex, counter++) = x;
						samples.at<float>(sampleIndex, counter++) = y;
					}
				}
			}

			if (pow(xc - x, 2) + pow(yc - y, 2) <= pow(rc, 2))
				sampleIndex++;


		}
	}

	Mat labels, centers;
	int attempts = 3;
	kmeans(samples, K, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


	Mat new_image(input_image.size(), CV_8UC3, Scalar(255,255,255));


	sampleIndex = 0;
	for (int y = 0; y < input_image.rows; y++) {
		for (int x = 0; x < input_image.cols; x++) {
			if (pow(xc - x, 2) + pow(yc - y, 2) <= pow(rc, 2)) {
				int cluster_idx = labels.at<int>(sampleIndex++, 0);
				if (input_image.channels() == 4) {
					for (int i = 0; i < input_image.channels() - 1; i++) {
						new_image.at<Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
					}
				}
				else {
					new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
				}
			}
		}
	}

	imshow("clustered image", new_image);
	waitKey(0);


	Mat forNow; cvtColor(new_image, forNow, COLOR_BGR2GRAY);
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

				Mat currentColorRegion(new_image.rows, new_image.cols, CV_8UC3);
				for (int y_ = 0; y_ < forNow.rows; y_++)
				{
					for (int x_ = 0; x_ < forNow.cols; x_++)
					{
						if (forNow.at<uchar>(y_, x_) == uc)
						{
							currentColorRegion.at<Vec3b>(y_, x_)[0] = new_image.at<Vec3b>(y_, x_)[0];
							currentColorRegion.at<Vec3b>(y_, x_)[1] = new_image.at<Vec3b>(y_, x_)[1];
							currentColorRegion.at<Vec3b>(y_, x_)[2] = new_image.at<Vec3b>(y_, x_)[2];
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
	// Somma i valori corrispondenti degli elementi dei vettori
	for (int r = 0; r < regions.size(); r++)
		imshow(to_string(r), regions.at(r));
	waitKey(0);

}



int main()
{
	//FOR SEGMENTING I NEED CIRCLES: THEIR CENTER'S COORDS AND RADIUS
	Mat img1Color = imread("leftover0_1.jpg");
	Mat img1Gray = imread("leftover0_1.jpg", IMREAD_GRAYSCALE);
	resize(img1Gray, img1Gray, Size(img1Gray.cols / 2, img1Gray.rows / 2));
	resize(img1Color, img1Color, Size(img1Color.cols / 2, img1Color.rows / 2));
	Mat imgCircles = img1Gray.clone();
	GaussianBlur(imgCircles, imgCircles, Size(9, 9), 2, 2);

	vector <Vec3f> circles;
	HoughCircles(img1Gray, circles, HOUGH_GRADIENT, 1.65, imgCircles.rows / 4, 160, 160, imgCircles.rows / 5, imgCircles.rows / 2);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// DRAW THE CIRCLE CENTER
		circle(imgCircles, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// DRAW THE CIRCLE OUTLINE
		circle(imgCircles, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	//imshow("Colored circles", imgCircles);
	//waitKey(0);

	//FOR EACH PLATE RECOGNIZED
	for (int c = 0; c < circles.size(); c++)
	{
		Vec3f circle_ = circles.at(c);
		int x_center = cvRound(circle_[0]);
		int y_center = cvRound(circle_[1]);
		int radius = cvRound(circle_[2]);

		Mat screenshot = Mat(img1Color.rows, img1Color.cols, CV_8UC4, Scalar(255, 255, 255,0));
		vector<Point> seeds;

		for (int y = 0; y < imgCircles.rows; y++)
		{
			for (int x = 0; x < imgCircles.cols; x++)
			{
				if (pow(x_center - x, 2) + pow(y_center - y, 2) <= pow(radius, 2))
				{
					screenshot.at<Vec4b>(y, x)[0] = img1Color.at<Vec3b>(y, x)[0];
					screenshot.at<Vec4b>(y, x)[1] = img1Color.at<Vec3b>(y, x)[1];
					screenshot.at<Vec4b>(y, x)[2] = img1Color.at<Vec3b>(y, x)[2];
					screenshot.at<Vec4b>(y, x)[3] = 255;
				}
			}
		}


		Mat screenshot_before = screenshot.clone();
		mySeg(screenshot,x_center,y_center,radius);
		imshow("screenshot_bef", screenshot_before);
		imshow("screenshot", screenshot);
		waitKey(0);

	}

	return 0;
}