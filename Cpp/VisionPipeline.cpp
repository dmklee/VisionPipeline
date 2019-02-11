#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"
#include "ContourDetection.hpp"

using namespace cv;
//g++ $(pkg-config --cflags --libs opencv) -w VisionPipeline.cpp -o VisionPipeline

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/occlusion2.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );
  // cv::Mat noise = Mat(image_gray.size(), CV_64F);
  // cv::randn(noise, 0, 0.05);
  // image_gray += noise;
  // normalize(image_gray, image_gray, 0.0, 1.0, CV_MINMAX, CV_64F);
  // image_gray.convertTo(image_gray, CV_32F, 255, 0);
  // suppressNoise(image, image);
  Mat BW = Mat(image.rows, image.cols, CV_8UC1);
  Mat RG = Mat(image.rows, image.cols, CV_8UC1);
  Mat YB = Mat(image.rows, image.cols, CV_8UC1);
  convertToBWRGYB(image, BW, RG, YB);
  // addGaussianNoise(BW, BW, 0, 30);
  // blur(BW, cv::Size(19,19), 10.0);
  // runEdgeDrawing(image_gray);
  // runLineDrawing(image_gray);
  extractContours(BW);

  // namedWindow("Line Map", WINDOW_NORMAL );
  // resizeWindow("Line Map", 800, 600);
  // imshow("Line Map", BW );
  // waitKey(0);
  return 0;
}
