#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <ctime>

typedef std::array<int,2> pt_type;
typedef std::vector< pt_type > seg_type;
typedef std::vector< seg_type > segList_type;


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

void extractAnchors(Mat& grad,Mat& dirMap, seg_type& anchorList, int gradThreshold,
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

int edgelWalkUp(int x, int y, Mat& grad, Mat& dirMap,
    Mat& edgeMap, seg_type& edgeSeg) {
  int last_move = 0;
  do {
    edgeMap.at<uchar>( x , y ) = 255;
    if ((grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x-1,y)) and
          (grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x-1,y+1))) {
      --x; --y;
      last_move = -1;
    } else if ((grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x-1,y)) and
          (grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x-1,y-1))) {
      --x; ++y;
      last_move = 1;
    } else{
      --x;
    }
    edgeSeg.push_back({x,y});
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad,x,y) and !isOccupied(edgeMap,x,y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkDown(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, seg_type& edgeSeg) {
  int last_move = 0;
  do {
    edgeMap.at<uchar>( x , y ) = 255;
    if ((grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x+1,y)) and
          (grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x+1,y+1))) {
      ++x; --y;
      last_move = -1;
    } else if ((grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x+1,y)) and
          (grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x+1,y-1))) {
      ++x; ++y;
      last_move = 1;
    } else{
      ++x;
    }
    edgeSeg.push_back({x,y});
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkLeft(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, seg_type& edgeSeg) {
  int last_move = 0;
  do {
    edgeMap.at<uchar>( x , y ) = 255;
    if ((grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x,y-1)) and
          (grad.at<uchar>(x-1,y-1) > grad.at<uchar>(x+1,y-1))) {
      --x; --y;
      last_move = -1;
    } else if ((grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x-1,y-1)) and
          (grad.at<uchar>(x+1,y-1) > grad.at<uchar>(x,y-1))) {
      ++x; --y;
      last_move = 1;
    } else{
      --y;
    }
    edgeSeg.push_back({x,y});
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkRight(int x, int y, Mat& grad, Mat& dirMap,
                    Mat& edgeMap, seg_type& edgeSeg) {
  int last_move = 0;
  do {
    edgeMap.at<uchar>( x , y ) = 255;
    if ((grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x,y+1)) and
          (grad.at<uchar>(x-1,y+1) > grad.at<uchar>(x+1,y+1))) {
      --x; ++y;
      last_move = -1;
    } else if ((grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x-1,y+1)) and
          (grad.at<uchar>(x+1,y+1) > grad.at<uchar>(x,y+1))) {
      ++x; ++y;
      last_move = 1;
    } else{
      ++y;
    }
    edgeSeg.push_back({x,y});
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

void sortAnchors(seg_type& anchorList, Mat& grad, seg_type& dst) {
  // return a list going from highest to lowest grad value
  std::copy(anchorList.begin(),anchorList.end(),std::back_inserter(dst));
  return;
}

int growSegment(Mat& grad, Mat& dirMap, Mat& edgeMap, pt_type& start,
                seg_type& seg, bool was_Horizontal, int move_id) {
  if (was_Horizontal) {
    if (move_id == 1) {
      return edgelWalkDown(start[0],start[1], grad, dirMap, edgeMap, seg);
    } else if (move_id == -1) {
      return edgelWalkUp(start[0],start[1], grad, dirMap, edgeMap, seg);
    } else {
      return 0;
    }
  } else {
    if (move_id == 1) {
      return edgelWalkRight(start[0],start[1],grad,dirMap,edgeMap, seg);
    } else if (move_id == -1) {
      return edgelWalkLeft(start[0],start[1],grad,dirMap,edgeMap, seg);
    } else {
      return 0;
    }
  }
}

void expandAnchor(Mat& grad, Mat& dirMap, Mat& edgeMap, int x, int y, seg_type& dst) {
  seg_type seg_A, seg_B;
  int move_A, move_B;
  bool was_Horizontal;
  edgeMap.at<uchar>(x,y) = 255;
  if (isHorizontal(dirMap, x , y)) {
    was_Horizontal = true;
    move_A = edgelWalkLeft(x,y,grad, dirMap, edgeMap, seg_A);
    move_B = edgelWalkRight(x,y, grad, dirMap, edgeMap, seg_B);
  } else {
    was_Horizontal = false;
    move_A = edgelWalkUp(x,y,grad, dirMap, edgeMap, seg_A);
    move_B = edgelWalkDown(x,y, grad, dirMap, edgeMap, seg_B);
  }
  while (move_A != 0 and move_B != 0) {
    if (move_A != 0) {
      move_A = growSegment( grad, dirMap, edgeMap, seg_A.back(), seg_A,
                            was_Horizontal, move_A);
    }
    if (move_B != 0) {
      move_B = growSegment( grad, dirMap, edgeMap, seg_B.back(), seg_B,
                            was_Horizontal, move_B);
    }
    was_Horizontal = !was_Horizontal;
  }
  dst.insert(dst.end(),seg_A.rbegin(), seg_A.rend());
  dst.push_back({x,y});
  dst.insert(dst.end(),seg_B.begin(),seg_B.end());
}

void expandAnchors(Mat& grad, Mat& dirMap, Mat& edgeMap, seg_type& anchorList,
                    segList_type& dst, int sizeThreshold = 3) {
  seg_type new_edge;
  for (const auto& anchor: anchorList) {
    if (!isOccupied(edgeMap, anchor[0], anchor[1])) {
      expandAnchor(grad,dirMap,edgeMap, anchor[0], anchor[1], new_edge);
      if (static_cast<int>(new_edge.size()) > sizeThreshold) {
        dst.push_back(new_edge);
      }
      new_edge.clear();
    }
  }

}

void findEdgeSegments(Mat& grad, Mat& dirMap, Mat& edgeMap, segList_type& dst) {
  seg_type anchorList;
  clock_t t = clock();
  extractAnchors(grad, dirMap, anchorList, 36, 8, 4);
  t = clock()-t;
  std::printf("I found %i anchors in %f ms.\n", static_cast<int>(anchorList.size()),
              ((float)t)/CLOCKS_PER_SEC *1000.0);
  seg_type anchorListSorted;
  sortAnchors(anchorList, grad, anchorListSorted);
  t = clock();
  expandAnchors(grad, dirMap, edgeMap, anchorListSorted, dst);
  t = clock()-t;
  std::printf("I produced %i edge segments in %f ms.\n", static_cast<int>(dst.size()),
              ((float)t)/CLOCKS_PER_SEC*1000.0);
}

#endif
