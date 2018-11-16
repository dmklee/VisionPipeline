#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/features_BW.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );
  runEdgeDrawing(image_gray);
  // runLineDrawing(image_gray);

  // namedWindow("Line Map", WINDOW_NORMAL );
  // resizeWindow("Line Map", 1000, 800);
  // imshow("Line Map", image_gray );
  // waitKey(0);
  return 0;
}
