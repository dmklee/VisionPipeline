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

bool isHorizontal(Mat& dirMap, int x, int y) {
  // edge is horizontal here
  return dirMap.at<uchar>( x , y ) == 0;
}

bool isOccupied(Mat& edgeMap, int x, int y) {
  return edgeMap.at<uchar>( x , y ) > 0;
}

bool isValidEdgel(Mat& grad, int x, int y) {
  if (x < 0 or y < 0 or x >= grad.rows or y >= grad.cols) {
    std::printf("index error in isValidEdgel\n" );
    return false;
  }
  int threshold = 8;
  return grad.at<uchar>( x , y ) > threshold;
}

bool isAnchor(int x, int y, Mat& grad, Mat& dirMap,
              int gradThreshold, int anchorThreshold) {
  if (grad.at<uchar>( x , y ) < gradThreshold) {
    return false;
  }
  if (!isHorizontal(dirMap, x, y) ) {
    if (((grad.at<uchar>(x,y)-grad.at<uchar>(x,y-1)) >= anchorThreshold) and
          ((grad.at<uchar>(x,y)-grad.at<uchar>(x,y+1)) >= anchorThreshold)) {
      return true;
    }
  } else {
    if (((grad.at<uchar>(x,y)-grad.at<uchar>(x-1,y)) >= anchorThreshold) and
          ((grad.at<uchar>(x,y)-grad.at<uchar>(x+1,y)) >= anchorThreshold)) {
      return true;
    }
  return false;
  }
}

void extractAnchors(Mat& grad,Mat& dirMap, pt_list& anchorList, int gradThreshold,
                    int anchorThreshold,int scanInterval) {
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

void edgelWalkUp(int x, int y, Mat& grad, Mat& dirMap,
    Mat& edgeMap, pt_list& edgeSeg) {
  do {
    edgeMap.at<uchar>( x , y ) = 75;
    if ((grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x-1,y)) and
          (grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x-1,y+1))) {
      --x; --y;
    } else if ((grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x-1,y)) and
          (grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x-1,y-1))) {
      --x; ++y;
    } else{
      --x;
    }
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
}

void edgelWalkDown(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, pt_list& edgeSeg) {
  do {
    edgeMap.at<uchar>( x , y ) = 75;
    if ((grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x+1,y)) and
          (grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x+1,y+1))) {
      ++x; --y;
    } else if ((grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x+1,y)) and
          (grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x+1,y-1))) {
      ++x; ++y;
    } else{
      ++x;
    }
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
}

void edgelWalkLeft(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, pt_list& edgeSeg) {
  do {
    edgeMap.at<uchar>( x , y ) = 75;
    if ((grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x,y-1)) and
          (grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x+1,y-1))) {
      --x; --y;
    } else if ((grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x-1,y-1)) and
          (grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x,y-1))) {
      ++x; --y;
    } else{
      --y;
    }
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
}

void edgelWalkRight(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, pt_list& edgeSeg) {
  do {
    edgeMap.at<uchar>( x , y ) = 75;
    if ((grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x,y+1)) and
          (grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x+1,y+1))) {
      --x; ++y;
    } else if ((grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x-1,y+1)) and
          (grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x,y+1))) {
      ++x; ++y;
    } else{
      ++y;
    }
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
}

void sortAnchors(pt_list& anchorList, Mat& grad, pt_list& dst) {
  // return a list going from highest to lowest grad value
  std::copy(anchorList.begin(),anchorList.end(),back_inserter(dst));
  return;
}

void findEdgeSegments(Mat& grad, Mat& dirMap, Mat& edgeMap) {
  // Mat edgeMap = Mat(grad.rows,grad.cols,CV_8UC1, Scalar::all(0)); // should be all zeros with size of grad

  pt_list anchorList;
  extractAnchors(grad, dirMap, anchorList, 36, 8, 4);

  pt_list anchorListSorted;
  sortAnchors(anchorList, grad, anchorListSorted);
  std::printf("I found %i anchors.\n", static_cast<int>(anchorList.size()));
  for (const auto& anchor: anchorListSorted) {
    if (!isOccupied(edgeMap, anchor[0], anchor[1])) {
      if (isHorizontal(dirMap, anchor[0], anchor[1])) {
        pt_list seg1, seg2;
        edgelWalkLeft(anchor[0],anchor[1], grad, dirMap, edgeMap, seg1);
        edgelWalkRight(anchor[0],anchor[1], grad, dirMap, edgeMap, seg2);
      } else { // vertical
        pt_list seg1, seg2;
        edgelWalkUp(anchor[0],anchor[1], grad, dirMap, edgeMap, seg1);
        edgelWalkDown(anchor[0],anchor[1], grad, dirMap, edgeMap, seg2);
      }
      edgeMap.at<uchar>(anchor[0],anchor[1]) = 255;
    }
  }

}

// void expandAnchor() {
//   if (isHorizontal(dirMap, x , y)) {
//
//   }
// }

void expandAnchors(Mat& grad, Mat& dirMap, Mat& edgeMap);


#endif
