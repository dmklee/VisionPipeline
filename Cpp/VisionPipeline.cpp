#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/chairs.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );
  // runEdgeDrawing(image_gray);
  runLineDrawing(image_gray);

  return 0;
}
