#ifndef EDGE_DRAWING_INCLUDE
#define EDGE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <ctime>
#include <fstream>

typedef std::array<int,2> pt_type;
typedef std::vector< pt_type > seg_type;
typedef std::vector< seg_type > segList_type;


double suppressNoise(const Mat& img, Mat& dst, bool time_it=true) {
  clock_t t = clock();
  cv::Size ksize(5,5);
  double sigma = 1.0;
  cv::GaussianBlur(img,dst,ksize,sigma);
  t = clock() - t;
  if (time_it) {
  std::printf("\tI applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);
  }
  return static_cast<double>(t)/CLOCKS_PER_SEC*1000.0;
}

void computeGradAndDirectionMap(const Mat& img_gray, Mat& grad, Mat& dirMap) {
  Mat grad_x, grad_y;
  int ddepth = CV_16S;
  cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
  cv::convertScaleAbs( grad_x, grad_x, 0.25);
  cv::Sobel( img_gray, grad_y, ddepth, 0, 1, 3 );
  cv::convertScaleAbs( grad_y, grad_y, 0.25);
  cv::add(grad_x, grad_y, grad);
  dirMap = grad_x >= grad_y;
}

inline bool isHorizontal(const Mat& dirMap, const int x, int y) {
  // edge is horizontal here
  return dirMap.at<uchar>( x , y ) == 0;
}

inline bool isOccupied(const Mat& edgeMap, const int x, const int y) {
  return edgeMap.at<uchar>( x , y ) > 0;
}

inline bool isValidEdgel(const Mat& grad, const int x, const int y) {
  if (x < 0 or y < 0 or x >= grad.rows or y >= grad.cols) {
    return false;
  }
  int threshold = 8;
  return grad.at<uchar>( x , y ) > threshold;
}

bool isAnchor(const int x, const int y, const Mat& grad, const Mat& dirMap,
              const int gradThreshold, const int anchorThreshold) {
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

void extractAnchors(const Mat& grad, const Mat& dirMap, seg_type& anchorList,
                    const int gradThreshold, const int anchorThreshold,
                    const int scanInterval) {
  // iterate over every "scanInterval"-th row and column
  // if isAnchor(pixel) then add to anchorList
  for (int i = 1; i < (grad.rows-1); i += scanInterval) {
    for (int j = 1; j < (grad.cols-1); j += scanInterval) {
      if (isAnchor(i,j,grad,dirMap, gradThreshold, anchorThreshold)) {
        pt_type tmp= {i,j};
        anchorList.push_back(tmp);
      }
    }
  }
  return;
}

int edgelWalkUp(int x, int y, const Mat& grad, const Mat& dirMap,
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
    pt_type tmp= {x,y};
    edgeSeg.push_back(tmp);
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad,x,y) and !isOccupied(edgeMap,x,y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkDown(int x, int y, const Mat& grad, const Mat& dirMap,
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
    pt_type tmp= {x,y};
    edgeSeg.push_back(tmp);
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          !isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkLeft(int x, int y, const Mat& grad, const Mat& dirMap,
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
    pt_type tmp= {x,y};
    edgeSeg.push_back(tmp);
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

int edgelWalkRight(int x, int y, const Mat& grad, const Mat& dirMap,
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
    pt_type tmp= {x,y};
    edgeSeg.push_back(tmp);
  } while (isValidEdgel(grad, x, y) and !isOccupied(edgeMap,x,y) and
          isHorizontal(dirMap,x,y));
  if (isValidEdgel(grad, x, y) and !isOccupied(edgeMap, x, y)) {
    return last_move;
  } else {
    edgeSeg.pop_back();
    return 0;
  }
}

void sortAnchors(const seg_type& anchorList, const Mat& grad, seg_type& dst) {
  // return a list going from highest to lowest grad value
  std::copy(anchorList.begin(),anchorList.end(),std::back_inserter(dst));
  return;
}

int growSegment(const Mat& grad, const Mat& dirMap, Mat& edgeMap, const pt_type& start,
                seg_type& seg, const bool was_Horizontal, const int move_id) {
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

void expandAnchor(const Mat& grad, const Mat& dirMap, Mat& edgeMap,
                  const int x, const int y, seg_type& dst) {
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
  pt_type tmp= {x,y};
  dst.push_back(tmp);
  dst.insert(dst.end(),seg_B.begin(),seg_B.end());
}

void expandAnchors(const Mat& grad, const Mat& dirMap, Mat& edgeMap, const seg_type& anchorList,
                    segList_type& dst, const int sizeThreshold = 3) {
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

void makeEdgeSegMap(Mat& edgeSegMap, segList_type& edgeSegments) {
  for (const auto& seg: edgeSegments) {
    int i = 255;
    for (const auto& point :seg) {
      edgeSegMap.at<uchar>(point[0],point[1]) = i;
    }
  }
}

void findEdgeSegments(Mat& grad, Mat& dirMap, Mat& edgeMap, segList_type& dst,
                      int gradThreshold = 36, int anchorThreshold = 8,
                      int scanInterval=4, int minLineLength=8) {
  seg_type anchorList;
  clock_t t = clock();
  extractAnchors(grad, dirMap, anchorList, gradThreshold,anchorThreshold,scanInterval);
  t = clock()-t;
  std::printf("I found %i anchors in %f ms.\n", static_cast<int>(anchorList.size()),
              ((float)t)/CLOCKS_PER_SEC *1000.0);
  seg_type anchorListSorted;
  sortAnchors(anchorList, grad, anchorListSorted);
  t = clock();
  expandAnchors(grad, dirMap, edgeMap, anchorListSorted, dst, minLineLength);
  t = clock()-t;
  std::printf("I produced %i edge segments in %f ms.\n", static_cast<int>(dst.size()),
              ((float)t)/CLOCKS_PER_SEC*1000.0);
}

void writeEdgeSegmentsToFile(segList_type& edgeSegments) {
  std::ofstream myfile;
  myfile.open("edgeSegments.txt");
  for (const auto& edgeSegment: edgeSegments) {
    std::printf("edge segment has length of %i\n", static_cast<int>(edgeSegment.size()));
    for (const auto& point: edgeSegment) {
      myfile << "(" << point[0] << "," << point[1] << ")\n";
    }
    myfile << "------- end of edge segment ---------\n";
  }
  myfile.close();
}

void runEdgeDrawing(Mat & image) {
  std::printf("Running Edge Drawing Algorithm...\n");
  std::printf("Image is %i by %i.\n", image.rows, image.cols);
  Mat grad, dirMap;
  clock_t t = clock();
  suppressNoise(image,image);
  t = clock()-t;
  std::printf("I applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  t = clock();
  computeGradAndDirectionMap(image,grad,dirMap);
  t = clock()-t;
  std::printf("I computed the gradient in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  seg_type anchorList;
  Mat anchorMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  const int gradThreshold = 5;
  const int anchorThreshold = 3;
  const int scanInterval = 2;
  extractAnchors(grad,dirMap,anchorList,gradThreshold, anchorThreshold, scanInterval);
  for (const auto& point: anchorList) {
    anchorMap.at<uchar>(point[0],point[1]) = 255;
  }

  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  segList_type edgeSegments;
  findEdgeSegments(grad, dirMap, edgeMap, edgeSegments);

  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  makeEdgeSegMap(edgeSegMap, edgeSegments);

  // namedWindow("Anchor Map", WINDOW_NORMAL );
  // resizeWindow("Anchor Map", 1000, 800);
  // imshow("Anchor Map", anchorMap);
  // waitKey(0);
  namedWindow("Edge Map", WINDOW_NORMAL );
  resizeWindow("Edge Map", 1000, 800);
  imshow("Edge Map", anchorMap );
  waitKey(0);
}

#endif
