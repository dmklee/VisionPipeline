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
}

void blur(Mat& img_gray, Size ksize, double sigma) {
  GaussianBlur(img_gray, img_gray, ksize, sigma);
}

void addGaussianNoise(Mat& img, Mat& dst, double mean, double stddev) {
  cv::Mat noise = Mat::zeros(img.size(), img.type());
  cv::randn(noise, 122+mean, stddev);
  dst += noise-122-mean;
}

void convertToBWRGYB(Mat& img_color, Mat& BW, Mat& RG, Mat& YB) {
  int b,g,r;
  for (int i =0; i < img_color.rows; i++) {
    for (int j=0; j < img_color.cols; j++) {
      b = img_color.at<Vec3b>(i,j)[0];
      g = img_color.at<Vec3b>(i,j)[1];
      r = img_color.at<Vec3b>(i,j)[2];
      BW.at<uchar>(i,j) = (b+g+r)/3;
      RG.at<uchar>(i,j) = (r - g + 255)/2;
      YB.at<uchar>(i,j) = ((r+g)/2 - b + 255)/2;
    }
  }
}

void ONcenterCell(Mat& img, Mat& dst) {
  Mat kernel = 2*getGaussianKernel(5,0.5,CV_32F) - getGaussianKernel(5,1.5,CV_32F);
  filter2D(img, dst, CV_16S, kernel);
  convertScaleAbs(dst, dst);
}

void OFFcenterCell(Mat& img, Mat& dst) {
  int data[9] = { 0, 1, 0, 1, -8, 1, 0, 1, 0};
  Mat kernel = Mat(3,3,CV_16S,data);
  filter2D(img, dst, CV_16S, kernel);
  convertScaleAbs(dst, dst, 0.25);
}

#endif
