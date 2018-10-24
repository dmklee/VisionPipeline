#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/ball_BW.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );

  Mat img;
  suppressNoise(image_gray, img);
  Mat grad,dirMap;
  computeGradAndDirectionMap(img, grad, dirMap);

  pt_list anchorList;
  double anchorThreshold = 8;
  int scanInterval = 4;
  extractAnchors(grad, dirMap, anchorList, anchorThreshold, scanInterval);

  Mat anchorMap = grad > 255;
  for ( const auto& point: anchorList) {
    anchorMap.at<ushort>(point[0],point[1]) = 255;
  }


  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", anchorMap);
  waitKey(0);
  // namedWindow("Display Image", WINDOW_AUTOSIZE );
  // imshow("Display Image", anchorMap);
  // waitKey(0);
  return 0;
}
