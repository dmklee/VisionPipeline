#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>

void suppressNoise(Mat& img_gray, Mat& dst) {
  cv::Size ksize = {5,5};
  double sigma = 1.0;
  cv::GaussianBlur(img_gray,dst,ksize,sigma);
}

void computeGradAndDirectionMap(Mat& img_gray, Mat& grad, Mat& dirMap) {
  Mat grad_x, grad_y;
  int ddepth = CV_16S;
  cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  cv::abs( grad_x );
  cv::Sobel( img_gray, grad_y, ddepth, -1, 0, 3 );
  cv::abs( grad_y );
  cv::add( grad_x, grad_y, grad );
  (grad_x >= grad_y).copyTo(dirMap);
}

void extractAnchors(Mat& grad){
  return;
}


#endif
