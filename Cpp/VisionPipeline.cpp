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

  Mat img, grad, dirMap;
  suppressNoise(image_gray,img);
  computeGradAndDirectionMap(img,grad,dirMap);

  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  edgeSeg_list edgeSegments;
  findEdgeSegments(grad, dirMap, edgeMap, edgeSegments);

  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  for (const auto& seg: edgeSegments) {
    int i = 255;
    for (const auto& point :seg) {
      edgeSegMap.at<uchar>(point[0],point[1]) = i;
      i--;
    }
    break;
  }

  // namedWindow("Direction Map", WINDOW_NORMAL );
  // resizeWindow("Direction Map", 1000, 800);
  // imshow("Direction Map", dirMap);
  // waitKey(0);
  namedWindow("Edge Map", WINDOW_NORMAL );
  resizeWindow("Edge Map", 1000, 800);
  imshow("Edge Map", edgeSegMap);
  waitKey(0);
  return 0;
}
