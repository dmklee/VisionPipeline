#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

void suppressNoise(Mat& img_gray, Mat& dst) {
  cv::Size ksize = {5,5};
  double sigma = 1.0;
  cv::GaussianBlur(img_gray,dst,ksize,sigma);
}

void computeGradAndDirectionMap(Mat& img_gray, Mat& grad, Mat& dirMap) {
  Mat grad_x, grad_y;
  int ddepth = cv:CV_16S;
  cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  cv::abs( grad_x );
  cv::Sobel( img_gray, grad_y, ddepth, -1, 0, 3 );
  cv::abs( grad_y );
  cv::add( grad_x, grad_y, grad );
  dirMap = grad_x >= grad_y;
}

bool isAnchor(int x, int y, Mat& grad, Mat& dirMap, double anchorThreshold) {
  if (dirMap.at<cv:CV_16S>(x,y) == 255 ) {
    //horizontal
    if (grad.at<double>(x,y)-grad.at<double>(x,y-1) >= anchorThreshold and
          grad.at<double>(x,y)-grad.at<double>(x,y+1) >= anchorThreshold) {
      return true;
    }
  } else {
    //vertical
    if (grad.at<double>(x,y)-grad.at<double>(x-1,y) >= anchorThreshold and
          grad.at<double>(x,y)-grad.at<double>(x+1,y) >= anchorThreshold) {
      return true;
    }
  return false;
  }
}

void extractAnchors(Mat& grad,Mat& dirMap,
                    std::vector<std::array<double,2> >& anchorList,
                    double anchorThreshold,int scanInterval){
  // iterate over every "scanInterval"-th row and column
  // if isAnchor(pixel) then add to anchorList
  int nrows;
  int ncols;
  for (int i = 0; i < nrows; i += scanInterval) {
    for (int j = 0; j < ncols; j += scanInterval) {
      if (isAnchor(i,j,grad,dirMap, anchorThreshold)) {
        anchorList.push_back({i,j});
      }
    }
  }
  return;
}



#endif
