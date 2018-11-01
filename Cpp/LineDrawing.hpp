#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <math.h>

typedef std::array<double,3> line_type; // A,B,C

void computeGrad(Mat& img, Mat& grad, Mat& dirMap, Mat& angleMap) {
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  int ddepth = CV_16S;
  cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  cv::convertScaleAbs( grad_x, abs_grad_x, 1);
  cv::Sobel( img_gray, grad_y, ddepth, 0, 1, 3 );
  cv::convertScaleAbs( grad_y, abs_grad_y, 1);
  dirMap = abs_grad_x >= abs_grad_y;
  cv::cartToPolar(grad_x,grad_y,grad,angleMap);
}


#endif
