#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/circle.png", 1 );
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );

  Mat edges;
  edgeDetector(image_gray,edges);

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", edges);
  waitKey(0);
  return 0;
}
