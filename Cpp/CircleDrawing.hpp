#ifndef CIRCLE_DRAWING_INCLUDE
#define CIRCLE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <math.h>

struct Arc {
  double _cX, _cY;
  double _radius;
  double _startAngle, _endAngle;
};

void leastSquaresCircleFit(lineChain_type& lineChain, Arc& arc, double& maxFitError ) {
  // https://dtcenter.org/met/users/docs/write_ups/circle_fit.pdf

  // find the mean values for x, y
  double x_bar, y_bar;
  int N;
  for (const auto& line: lineChain) {
    for (const auto& pt: line._data) {
      x_bar += pt[0];
      y_bar += pt[1];
      N++;
    }
  }
  x_bar /= N;
  y_bar /= N;

  double u, v;
  double Suu, Suv, Svv, Suuu, Svvv, Suvv, Svuu;
  Suu = Suv = Svv = Suuu = Svvv = Suvv = Svuu = 0.0;
  for (const auto& line: lineChain) {
    for (const auto& pt: line._data) {
      u = pt[0] - x_bar;
      v = pt[1] - y_bar;
      Suu  += u*u;
      Suv  += u*v;
      Svv  += v*v;
      Suuu += u*u*u;
      Svvv += v*v*v;
      Suvv += u*v*v;
      Svuu += v*u*u;
    }
  }
  double v_c, u_c, r2;
  v_c = 1/2*((Suuu + Suvv)/Suu - (Svvv+Svuu)/Suv) / (Suv/Suu - Svv/Suv);
  u_c = (1/2*(Suuu+Suvv) - Suv*v_c )/Suu;
  r2 = u_c*u_c + v_c*v_c + (Suu+Svv)/N;
  arc._cX = u_c + x_bar;
  arc._cY = v_c + y_bar;
  arc._radius = sqrt(r2);
  // find start and end angle
  error = 0.0;
  for (const auto& line: lineChain) {
    for (const auto& pt: line._data) {
      error += pow(pow(pt[0] - arc._cX,2) + pow(pt[1] - arc._cY,2) - r2 , 2 );
    }
  }
  error = sqrt(error/N);
}


void runCircleDrawing(Mat& img) {
  std::printf("Running Circle Drawing algorithm...\n" );
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

  Mat lineMap;
  cvtColor(edgeSegMap, lineMap, cv::COLOR_GRAY2BGR);
  makeLineMap(lineChains, lineMap);

  namedWindow("Line Map", WINDOW_NORMAL );
  resizeWindow("Line Map", 1000, 800);
  imshow("Line Map", lineMap );
  waitKey(0);
}




#endif
