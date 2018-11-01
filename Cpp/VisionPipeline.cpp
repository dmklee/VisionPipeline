#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/ellipses.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );

  Mat img;
  suppressNoise(image_gray, img);
  Mat grad, dirMap;
  computeGradAndDirectionMap(img, grad, dirMap);

  // pt_list anchorList;
  // int gradThreshold = 36;
  // int anchorThreshold = 8;
  // int scanInterval = 1;
  //
  // extractAnchors(grad, dirMap, anchorList, gradThreshold, anchorThreshold, scanInterval);
  //
  // Mat anchorMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  // for ( const auto& point: anchorList) {
  //   anchorMap.at<uchar>(point[0],point[1]) = 255;
  //   // std::printf("%i, %i\n", point[0],point[1] );
  // }
  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  findEdgeSegments(grad, dirMap, edgeMap);

  // namedWindow("Direction Map", WINDOW_NORMAL );
  // resizeWindow("Direction Map", 1000, 800);
  // imshow("Direction Map", dirMap);
  // waitKey(0);
  namedWindow("Edge Map", WINDOW_NORMAL );
  resizeWindow("Edge Map", 1000, 800);
  imshow("Edge Map", edgeMap);
  waitKey(0);
  return 0;
}
