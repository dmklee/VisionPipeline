#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <stdio.h>

typedef std::vector< std::array<int,2> > pt_list;
typedef std::vector< pt_list > edgeSeg_list;

void suppressNoise(Mat& img_gray, Mat& dst) {
  cv::Size ksize = {5,5};
  double sigma = 1.0;
  cv::GaussianBlur(img_gray,dst,ksize,sigma);
}

void computeGradAndDirectionMap(Mat& img_gray, Mat& grad, Mat& dirMap) {
  Mat grad_x, grad_y;
  int ddepth = CV_16S;
  cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  cv::convertScaleAbs( grad_x, grad_x, 0.25);
  cv::Sobel( img_gray, grad_y, ddepth, 0, 1, 3 );
  cv::convertScaleAbs( grad_y, grad_y, 0.25);
  cv::add(grad_x, grad_y, grad);
  dirMap = grad_x >= grad_y;
}

bool isAnchor(int x, int y, Mat& grad, Mat& dirMap,
              int gradThreshold, int anchorThreshold) {
  int gradThreshold = 36;
  gradThreshold *= 32767/255;
  anchorThreshold *= 32767/255;
  if (grad.at<ushort>(x,y) < gradThreshold) {
    return false;
  }
  if (dirMap.at<ushort>(x,y) != 0 ) {
    //horizontal
    if (((grad.at<ushort>(x,y)-grad.at<ushort>(x,y-1)) >= anchorThreshold) and
          ((grad.at<ushort>(x,y)-grad.at<ushort>(x,y+1)) >= anchorThreshold)) {
      return true;
    }
  } else {
    //vertical
    if (((grad.at<ushort>(x,y)-grad.at<ushort>(x-1,y)) >= anchorThreshold) and
          ((grad.at<ushort>(x,y)-grad.at<ushort>(x+1,y)) >= anchorThreshold)) {
      return true;
    }
  return false;
  }
}

void extractAnchors(Mat& grad,Mat& dirMap,
                    pt_list& anchorList, int gradThreshold,
                    int anchorThreshold,int scanInterval)
{
  // iterate over every "scanInterval"-th row and column
  // if isAnchor(pixel) then add to anchorList
  for (int i = 1; i < (grad.rows-1); i += scanInterval) {
    for (int j = 1; j < (grad.cols-1); j += scanInterval) {
      if (isAnchor(i,j,grad,dirMap, gradThreshold, anchorThreshold)) {
        anchorList.push_back({i,j});
      }
    }
  }
  return;
}

void edgelWalkLeft(int x, int y, Mat& grad, Mat& dirMap, Mat& edgeMap,
                    pt_list& edgeSeg) {
  while (grad.at<ushort>(x,y) > 0 and edgeMap.at<ushort> == 0 and
          dirMap.at<ushort> != 0) {
    edgeMap.at<ushort> = 1;
    if ((grad.at<ushort>(x-1,y-1) > grad.at<ushort>(x-1,y)) and
          (grad.at<ushort>(x-1,y-1) > grad.at<ushort>(x-1,y+1))) {
      --x; --y;
    } else if ((grad.at<ushort>(x-1,y+1) > grad.at<ushort>(x-1,y)) and
          (grad.at<ushort>(x-1,y+1) > grad.at<ushort>(x-1,y-1))) {
      ++x; ++y;
    } else{
      --x;
    }
  }

}

void sortAnchors(pt_list& anchorList, Mat& grad, pt_list& dst) {
  // return a list going from highest to lowest grad value
  return;
}

void findEdgeSegments(Mat& grad, Mat& dirMap) {
  Mat edgeMap; // should be all zeros with size of grad

  pt_list anchorList;
  extractAnchors(grad, dirMap, anchorList, 36, 8);

  pt_list anchorListSorted;
  sortAnchors(anchorList, grad, anchorListSorted);

  for (const auto& anchor: anchorListSorted) {
    if (edgeMap.at<ushort>(anchor[0],anchor[1]) == 0) {
      if (horizontal) {
        pt_list seg1 = edgelWalkLeft
        pt_list seg2 = edgelWalkRight
      } else {
        pt_list seg1 = edgelWalkUp
        pt_list seg2 = edgelWalkDown
      }
    }
  }

}

void expandAnchors(Mat& grad, Mat& dirMap, Mat& edgeMap,
                    )


#endif
