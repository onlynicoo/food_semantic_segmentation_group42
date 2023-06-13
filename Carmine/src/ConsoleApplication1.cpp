#include "mySegmentation.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main()
{
	//FOR SEGMENTING I NEED CIRCLES: THEIR CENTER'S COORDS AND RADIUS
    Mat img1Color = imread("leftover0_1.jpg");
	Mat img1Gray = imread("leftover0_1.jpg", IMREAD_GRAYSCALE);
	resize(img1Gray, img1Gray, Size(img1Gray.cols/2, img1Gray.rows/2));
    resize(img1Color, img1Color, Size(img1Color.cols / 2, img1Color.rows / 2));
	Mat imgCircles = img1Gray.clone();
	GaussianBlur(imgCircles, imgCircles, Size(9, 9), 2, 2);
	
	vector <Vec3f> circles;
	HoughCircles(img1Gray, circles, HOUGH_GRADIENT, 1.65, imgCircles.rows / 4, 160, 160,imgCircles.rows/5, imgCircles.rows/2);
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

		Mat screenshot = Mat(img1Color.rows, img1Color.cols, CV_8UC3, Scalar(255,255,255));
        vector<Point> seeds;

		for (int y = 0; y < imgCircles.rows; y++)
		{
			for (int x = 0; x < imgCircles.cols; x++)
			{
                if (pow(x_center - x, 2) + pow(y_center - y, 2) <= pow(radius, 2))
                {
                    screenshot.at<Vec3b>(y, x)[0] = img1Color.at<Vec3b>(y, x)[0];
                    screenshot.at<Vec3b>(y, x)[1] = img1Color.at<Vec3b>(y, x)[1];
                    screenshot.at<Vec3b>(y, x)[2] = img1Color.at<Vec3b>(y, x)[2];
                }
			}
		}


		Mat screenshot_before = screenshot.clone();
		mySeg(screenshot);
		imshow("screenshot_bef",screenshot_before);
		imshow("screenshot", screenshot);
        waitKey(0);

	}

	return 0;
}