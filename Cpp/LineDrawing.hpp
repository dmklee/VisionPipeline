#ifndef LINE_DRAWING_INCLUDE
#define LINE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <math.h>



struct LineSeg
{
  float _A,_B,_C;
  seg_type _data;
};

void computeGrad(Mat& img, Mat& grad, Mat& dirMap, Mat& angleMap) {
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  int ddepth = CV_32F;
  cv::Sobel( img, grad_x, ddepth, 1, 0, 3, 0.5);
  cv::convertScaleAbs( grad_x, abs_grad_x, 1);
  cv::Sobel( img, grad_y, ddepth, 0, 1, 3, 0.5 );
  cv::convertScaleAbs( grad_y, abs_grad_y, 1);
  dirMap = abs_grad_x >= abs_grad_y;
  cv::cartToPolar(grad_x,grad_y,grad,angleMap);
  grad.convertTo(grad,CV_8U);
}

int computeMinLineLength(Mat& img) {
  int N = (img.rows*img.cols)/2;
  int n = -4*log(N)/log(0.125);
  return n;
}

bool isAligned(Mat& angleMap, pt_type& pix_A, pt_type& pix_B) {
  float angle_A = angleMap.at<float>(pix_A[0],pix_A[1]);
  float angle_B = angleMap.at<float>(pix_B[0],pix_B[1]);
  float diff = abs(angle_A - angle_B);
  if (diff > M_PI) diff = 2*M_PI-diff;
  return  diff <= M_PI/8.0;
}

void runLineDrawing(Mat& img) {
  std::printf("Running Line Drawing algorithm...\n" );
  std::printf("Image size is %i by %i\n", static_cast<int>(img.rows),
              static_cast<int>(img.cols));
  clock_t t = clock();
  suppressNoise(img,img);
  t = clock()-t;
  std::printf("I applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  Mat grad,dirMap,angleMap;
  t = clock();
  computeGrad(img, grad, dirMap, angleMap);
  t = clock()-t;
  std::printf("I computed gradient and angle map in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);
  segList_type edgeSegments;
  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  int gradThreshold = 5;
  int anchorThreshold = 1;
  int scanInterval = 4;
  findEdgeSegments(grad, dirMap, edgeMap, edgeSegments,
                    gradThreshold, anchorThreshold, scanInterval);

  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  makeEdgeSegMap(edgeSegMap, edgeSegments);

  namedWindow("Grad Map", WINDOW_NORMAL );
  resizeWindow("Grad Map", 1000, 800);
  imshow("Grad Map", edgeSegMap);
  waitKey(0);
}
#endif
