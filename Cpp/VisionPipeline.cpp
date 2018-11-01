#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"

using namespace cv;

int main(int argc, char** argv )
{
  Mat image = imread("../Pics/chairs.png", 1);
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );
  std::printf("Image is %i by %i.\n", image.rows, image.cols);
  Mat img, grad, dirMap;
  clock_t t = clock();
  suppressNoise(image_gray,image_gray);
  t = clock()-t;
  std::printf("I applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  t = clock();
  computeGradAndDirectionMap(image_gray,grad,dirMap);
  t = clock()-t;
  std::printf("I computed the gradient in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  seg_type anchorList;
  Mat anchorMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  extractAnchors(grad,dirMap,anchorList,36,2,4);
  for (const auto& point: anchorList) {
    anchorMap.at<uchar>(point[0],point[1]) = 255;
  }

  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  segList_type edgeSegments;
  findEdgeSegments(grad, dirMap, edgeMap, edgeSegments);

  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  for (const auto& seg: edgeSegments) {
    int i = 255;
    for (const auto& point :seg) {
      edgeSegMap.at<uchar>(point[0],point[1]) = i;
    }
    i -= 70;
  }

  // namedWindow("Anchor Map", WINDOW_NORMAL );
  // resizeWindow("Anchor Map", 1000, 800);
  // imshow("Anchor Map", anchorMap);
  // waitKey(0);
  namedWindow("Edge Map", WINDOW_NORMAL );
  resizeWindow("Edge Map", 1000, 800);
  imshow("Edge Map", edgeSegMap );
  waitKey(0);
  return 0;
}
