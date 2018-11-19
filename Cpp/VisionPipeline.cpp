#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/toyblocks.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );
  Mat BW = Mat(image.rows,image.cols, CV_8UC1);
  Mat RG = Mat(image.rows,image.cols, CV_8UC1);
  Mat YB = Mat(image.rows,image.cols, CV_8UC1);
  convertToBWRGYB(image, BW, RG, YB);
  // runEdgeDrawing(image_gray);
  // runLineDrawing(image_gray);

  namedWindow("Line Map", WINDOW_NORMAL );
  resizeWindow("Line Map", 1000, 800);
  imshow("Line Map", RG );
  namedWindow("Line Map2", WINDOW_NORMAL );
  resizeWindow("Line Map2", 1000, 800);
  imshow("Line Map2", YB );
  waitKey(0);
  return 0;
}
