#ifndef TRANSFORMATIONS_INCLUDE
#define TRANSFORMATIONS_INCLUDE

#include <opencv2/opencv.hpp>

using namespace cv;

void edgeDetector(Mat& img_gray, Mat& edges, Mat& grad_y, Mat& grad_x) {
  Mat abs_grad_x, abs_grad_y;
  int ddepth = CV_16S;

  Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  convertScaleAbs( grad_x, abs_grad_x );

  Sobel( img_gray, grad_y, ddepth, 0, 1, 3);
  convertScaleAbs( grad_y, abs_grad_y );

  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges );
  return;
}

void edgeDetector(Mat& img_gray, Mat& edges) {
  Mat grad_x;
  Mat grad_y;
  edgeDetector(img_gray,edges,grad_x,grad_y);
}

void blur(Mat& img_gray, Mat& dst, Size ksize, double sigma) {
  GaussianBlur(img_gray,dst,ksize,sigma);
  return;
}

void applyGaussianFilter() {
  return;
}


#endif
