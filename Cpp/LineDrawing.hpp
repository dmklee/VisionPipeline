#ifndef LINE_DRAWING_INCLUDE
#define LINE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <math.h>

struct Line {
  // _A * x + _B * y + _C = 0
  float _A,_B,_C;
  seg_type _data; // may or may not be needed
};

typedef std::vector< Line > lineChain_type;
typedef std::vector< lineChain_type > lineChainList_type;
typedef std::vector< seg_type >::iterator seg_it_type;

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

void leastSquaresLineFit(seg_it_type& it, int minLineLength, Line& line,
                          double& error) {
  // decide if its more horizontal or vertical
  double est_slope = ((*(it+minLineLength-1))[1] - (*(it+1))[1]) /
                      ((*(it+minLineLength-1))[0] - (*(it+1))[0]);
  bool isVertical = abs(est_slope) > 1000;
  int sum_x, sum_y, sum_xy, sum_x2, sum_y2;
  sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0;
  for (int i=0; i<minLineLength; i++) {
    sum_x += (*(it+i))[0];
    sum_y += (*(it+i))[1];
    sum_xy += (*(it+i))[0]*(*(it+i))[1];
    sum_x2 += pow((*(it+i))[0],2);
    sum_y2 += pow((*(it+i))[1],2);
  }
  if (isVertical) {
    std::swap(sum_x,sum_y);
    std::swap(sum_x2,sum_y2);
  }
  int c = (minLineLength+1)*pow(sum_x2-sum_x,2);
  double a,b;
  a = 1/c * ((minLineLength+1)*sum_xy-sum_x*sum_y);
  b = 1/c * (sum_x2*sum_y - sum_x*sum_xy);
  if (isVertical) {
    line._A = 1.0;
    line._B = -a;
    line._C = b;
  } else {
    line._A = a;
    line._B = -1.0;
    line._C = b;
  }
  // now compute the error
  error = 0.0;
  for (int i=0; i < minLineLength; ++i) {
    error += pow(line._A*(*(it+i))[0] + line._B*(*(it+i))[1] +line._C, 2.0);
  }
}

double computePointDistance2Line(Line& line, const seg_it_type& it) {
  return abs(line._A*(*it)[0] + line._B*(*it)[1] + line._C)/
          std::sqrt(pow(line._A,2)+pow(line._B,2));
}

void lineFit(seg_it_type& it, size_t numPixels, lineChain_type& dst,
            int minLineLength, int maxFitError) {
  // return true unless out of pixels
  double error = 10*lineFitError;
  Line line;
  while (numPixels > minLineLength) {
    leastSquaresLineFit(it, minLineLength, line, error);
    if (error <= maxFitError) break; // initial line detected
    it++;
    numPixels--;
  }
  if (error > maxFitError) return; // out of pixels

  for (int i=0; i != minLineLength; ++i) {
    line._data.push_back({*(it+i)[0],*(it+i)[1]});
  }

  int lineLength = minLineLength;
  double d;
  while (lineLength < numPixels) {
    d = computePointDistance2Line(line, it+lineLength);
    if (d > maxFitError) break;
    line._data.push_back({*(it+lineLength)[0],*(it+lineLength)[1]});
    lineLength++;
  }
  it += lineLength;
}

void generateLines(segList_type edgeSegments, lineChainList_type dst,
                    int minLineLength, int maxFitError=1) {
  lineChain_type lineChain;
  seg_it_type it;
  size_t segLength;
  for (const auto& edgeSegment: edgeSegments) {
    it = edgeSegment.begin();
    segLength = edgeSegment.end()-it;
    while (segLength > minLineLength) {
      lineFit(it, segLength, lineChain, minLineLength, maxFitError));
    }
    if (!lineChain.empty()) {
      dst.push_back(lineChain);
      lineChain.clear();
    }
  }
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

  lineList_type lineList;
  generateLines(edgeSegments, lineList);


  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  makeEdgeSegMap(edgeSegMap, edgeSegments);




  namedWindow("Grad Map", WINDOW_NORMAL );
  resizeWindow("Grad Map", 1000, 800);
  imshow("Grad Map", edgeSegMap);
  waitKey(0);
}
#endif
