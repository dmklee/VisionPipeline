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
  double _sumX, _sumY, _sumXY, _sumX2;
  seg_type _data; // may or may not be needed
};

typedef std::vector< Line > lineChain_type;
typedef std::vector< lineChain_type > lineChainList_type;
typedef seg_type::iterator seg_it_type;

void computeGrad(const Mat& img, Mat& grad, Mat& dirMap, Mat& angleMap) {
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

int computeMinLineLength(const Mat& img) {
  int N = pow(img.rows*img.cols, 0.5);
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

void leastSquaresLineFit(const seg_it_type& it, const int minLineLength,
                          Line& L, double& error) {

  int x = (*(it+minLineLength))[0];
  int y = (*(it+minLineLength))[1];
  L._sumX += x;
  L._sumY += y;
  L._sumXY += x*y;
  L._sumX2 += x*x;
  double xMean = L._sumX / minLineLength;
  double yMean = L._sumY / minLineLength;
  double denom = L._sumX2 - L._sumX * xMean;
  if (std::fabs(denom) < 1e-7) {
    //vertical line
    L._A = 1.0;
    L._B = 0.0;
    L._C = -xMean;
  } else {
    L._A = -(L._sumXY - L._sumX * yMean) / denom;
    L._B = 1.0;
    L._C = - (yMean + L._A * xMean);
  }
  // now compute the root mean square error
  error = 0.0;
  for (int i=0; i < minLineLength; ++i) {
    error += pow(L._A*(*(it+i))[0] + L._B*(*(it+i))[1] +L._C, 2.0);
  }
  error = sqrt(error/minLineLength);

  x = (*it)[0];
  y = (*it)[1];
  L._sumX  -= x;
  L._sumY  -= y;
  L._sumXY -= x*y;
  L._sumX2 -= x*x;
}

inline double computePointDistance2Line(const Line& L, const seg_it_type& it) {
  return abs(L._A * (*it)[0] + L._B * (*it)[1] + L._C)/
          std::sqrt(pow(L._A,2)+pow(L._B,2));
}

void lineFit(seg_it_type& it, int numPixels, lineChain_type& lineChain,
             const int minLineLength, const float maxFitError) {
  // return true unless out of pixels
  double error = 10*maxFitError;
  Line L;
  L._sumX = L._sumY = L._sumXY = L._sumX2 = 0.0;
  int x,y;
  for (int i=0; i<(minLineLength-1); i++) {
    x = (*(it+i))[0];
    y = (*(it+i))[1];
    L._sumX += x;
    L._sumY += y;
    L._sumXY += x*y;
    L._sumX2 += x*x;
  }
  while (numPixels > minLineLength) {
    leastSquaresLineFit(it, minLineLength, L, error);
    if (error <= maxFitError) {
      break;
    } // initial line detected
    it++;
    numPixels--;
  }
  if (error > maxFitError) return; // out of pixels

  for (int i=0; i != minLineLength; ++i) {
    pt_type tmp= {(*(it+i))[0],(*(it+i))[1]};
    L._data.push_back(tmp);
  }

  int lineLength = minLineLength;
  it += lineLength;
  double d;
  while (lineLength < numPixels) {
    d = computePointDistance2Line(L, it);
    if (d > maxFitError) break;
    pt_type tmp= {(*(it))[0],(*(it))[1]};
    L._data.push_back(tmp);
    lineLength++;
    it++;
  }
  lineChain.push_back(L);
}

void generateLines(segList_type& edgeSegments, lineChainList_type& dst,
                    const int minLineLength=12, const float maxFitError=2) {
  lineChain_type lineChain;
  // seg_it_type it;
  // int segLength;
  for (auto& edgeSegment: edgeSegments) {
    seg_it_type it = edgeSegment.begin();
    int segLength = edgeSegment.end()-it;
    while (segLength > minLineLength) {
      lineFit(it, segLength, lineChain, minLineLength, maxFitError);
      segLength = edgeSegment.end()-it;
    }
    if (!lineChain.empty()) {
      dst.push_back(lineChain);
      lineChain.clear();
    }
  }
}

void makeLineMap(lineChainList_type& lineChainList, Mat& lineMap) {
  Scalar color(0,0,255);
  int thickness = 1;
  int x1, y1, x2, y2;
  for (const auto& lineChain: lineChainList) {
    for (const auto& L: lineChain) {
      x1 = L._data.front()[0]; y1 = L._data.front()[1];
      x2 = L._data.rbegin()[2][0]; y2 = L._data.rbegin()[2][1];
      cv::line ( lineMap, Point(y1, x1), Point(y2, x2), color, thickness);
    }
  }

}

void runLineDrawing(Mat& img) {
  std::printf("Running Line Drawing algorithm...\n" );
  std::printf("Image size is %i by %i\n", static_cast<int>(img.rows),
              static_cast<int>(img.cols));
  clock_t t = clock();
  float total = 0;
  suppressNoise(img,img);
  t = clock()-t;
  std::printf("I applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  Mat grad,dirMap;
  t = clock();
  computeGradAndDirectionMap(img, grad, dirMap);
  t = clock()-t;
  std::printf("I computed gradient and angle map in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);
  segList_type edgeSegments;
  Mat edgeMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  int gradThreshold = 5;
  int anchorThreshold = 3;
  int scanInterval = 4;
  int minLineLength = computeMinLineLength(grad);
  findEdgeSegments(grad, dirMap, edgeMap, edgeSegments, gradThreshold,
                  anchorThreshold, scanInterval, minLineLength);

  Mat edgeSegMap = Mat::zeros(grad.rows,grad.cols,CV_8U);
  makeEdgeSegMap(edgeSegMap, edgeSegments);

  t = clock();
  lineChainList_type lineChains;
  generateLines(edgeSegments, lineChains, minLineLength);
  t = clock()-t;
  int numLines = 0;
  for (const auto& lineChain: lineChains) {
    numLines += lineChain.size();
  }
  std::printf("I generated %i line segments in %f ms\n", numLines,
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  Mat lineMap = Mat::zeros(grad.rows,grad.cols, CV_8U);
  cvtColor(lineMap, lineMap, cv::COLOR_GRAY2BGR);
  makeLineMap(lineChains, lineMap);

  namedWindow("Line Map", WINDOW_NORMAL );
  resizeWindow("Line Map", 1000, 800);
  imshow("Line Map", lineMap );
  waitKey(0);
}

#endif
