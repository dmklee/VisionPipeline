#ifndef CONTOURDETECTION_INCLUDE
#define CONTOURDETECTION_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include <algorithm>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
// #include <Eigen/Dense>

#define PI 3.14159265

using namespace cv;
// using namespace Eigen;

typedef std::vector<cv::Point>::const_iterator vec_iter_t;

inline int getGradID(const short grad_x, const short grad_y) {
  float ratio = grad_x != 0 ? ((float)grad_y)/grad_x : ((float)grad_y)/0.0001;
  if (abs(ratio) < 0.5) {
    if (grad_x >= 0) return 0;
    return 4;
  }
  if (ratio > 0.5 & ratio < 2.0) {
    if (grad_x >= 0) return 7;
    return 3;
  }
  if (ratio < -0.5 & ratio > -2.0) {
    if (grad_x >= 0) return 1;
    return 5;
  }
  if (grad_y < 0) return 2;
  return 6;
}

inline int getGradID(const Mat gradMap[], const Point& pt) {
  const short grad_x = gradMap[0].at<short>(pt.x, pt.y);
  const short grad_y = gradMap[1].at<short>(pt.x, pt.y);
  return getGradID(grad_x, grad_y);
}

inline int subtractGradID_abs(const int& g1, const int& g2) {
  int diff = abs(g1 - g2);
  if (diff > 4) diff = 8 - diff;
  return diff;
}

// inline int subtractGradID(const int& g1, const int& g2) {
//   std::printf("subtractGradID not working\n" );
//   int diff = abs(g1 - g2);
//   if (diff > 4) diff = 8 - diff;
//   return diff;
// }

double computeEdgeAndGradMap(Mat& image_gray, Mat& edgeMap, Mat gradMap[], bool time_it) {
  clock_t t = clock();
  Mat abs_grad_x, abs_grad_y;
  int ddepth = CV_16S;
  cv::Sobel( image_gray, gradMap[0], ddepth, 1, 0, 3 );
  cv::Sobel( image_gray, gradMap[1], ddepth, 0, 1, 3 );
  cv::convertScaleAbs( gradMap[0], abs_grad_x, 0.4);
  cv::convertScaleAbs( gradMap[1], abs_grad_y, 0.4);
  cv::add(abs_grad_x, abs_grad_y, edgeMap);
  t = clock() - t;
  if (time_it) {
  std::printf("\tI computed edge and gradient maps in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);
  }
  return static_cast<double> (t) / CLOCKS_PER_SEC*1000.0;
}

inline void moveAlongContour(cv::Point& new_pt, const int& grad_id,
                              const Mat& edgeMap) {
    static cv::Point offsets[] = {Point(-1,0), Point(-1,-1), Point(0,-1),
                                  Point(1,-1), Point(1,0), Point(1,1),
                                  Point(0,1), Point(-1,1)};

    int max_val = edgeMap.at<uchar>(new_pt.x+offsets[grad_id].x, new_pt.y+offsets[grad_id].y);
    int max_id = grad_id;
    int ret = 0;
    if (max_val < edgeMap.at<uchar>(new_pt.x+offsets[(grad_id+1) % 8].x,
                                    new_pt.y+offsets[(grad_id+1) % 8].y)) {
        max_id = (grad_id+1) % 8;
        ret = 1;
    } else if (max_val < edgeMap.at<uchar>(new_pt.x+offsets[(grad_id+7) % 8].x,
                                    new_pt.y+offsets[(grad_id+7) % 8].y)) {
        max_id = (grad_id+7) % 8;
        ret = -1;
    }
    new_pt += offsets[max_id];
}

inline bool isValidSeed( const int x, const int y, const Mat& edgeMap, const Mat gradMap[],
                 const int gradThreshold=10) {
  if (edgeMap.at<uchar>(x,y) < gradThreshold) return false;
  int grad_id = getGradID(gradMap, Point(x,y)) % 4;
  switch (grad_id) {
    case 0:  if (edgeMap.at<uchar>(x,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x,y-1) < gradThreshold) {
                        return false;
                      } break;
    case 1:      if (edgeMap.at<uchar>(x-1,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x+1,y-1) < gradThreshold) {
                        return false;
                      } break;
    case 2:    if (edgeMap.at<uchar>(x+1,y) < gradThreshold or
                          edgeMap.at<uchar>(x-1,y) < gradThreshold) {
                        return false;
                      } break;
    case 3:    if (edgeMap.at<uchar>(x+1,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x-1,y-1) < gradThreshold) {
                        return false;
                      } break;
  }

  return true;
}

inline bool shiftSeed(Point& seed, const Mat& edgeMap, const Mat gradMap[]) {
  int grad_id  = getGradID(gradMap, seed) % 4;
  uchar edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
  switch (grad_id) {
    case 0:  while (edgeVal < edgeMap.at<uchar>(seed.x, seed.y+1)) {
                      seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x, seed.y-1)) {
                      seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case 3:  while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y-1)) {
                      seed.x--; seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while(edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y+1)) {
                      seed.x++; seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case 2:  while (edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y)) {
                      seed.x++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y)) {
                      seed.x--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case 1:  while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y+1)) {
                      seed.x--; seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y-1)) {
                      seed.x++; seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
  }
  return edgeMap.at<uchar>(seed.x, seed.y) > 15;
}

inline bool isStableSeed(const Point& seed, const Mat& edgeMap,
          const Mat gradMap[], const int range=3 ) {
  Point pt = Point(seed);
  int center_grad_id = getGradID(gradMap, seed);
  int left_grad_id, right_grad_id;

  for (int i=0; i < range; i++) {
    left_grad_id = getGradID(gradMap, pt);
    moveAlongContour(pt, left_grad_id, edgeMap);
  }


  pt = Point(seed);
  for (int i=0; i < range; i++) {
    right_grad_id = getGradID(gradMap, pt);
    moveAlongContour(pt, right_grad_id, edgeMap);

  }


  return ( (subtractGradID_abs(center_grad_id, left_grad_id) < 2) &&
           (subtractGradID_abs(center_grad_id, right_grad_id) < 2) &&
           (subtractGradID_abs(right_grad_id, left_grad_id) < 2) );
}

double extractSeeds(Mat& edgeMap, Mat gradMap[], std::vector<Point>& dst,
                  bool time_it, int size=15, int threshold = 15) {
  clock_t t = clock();
  Mat region_frame;
  Point current_pt;
  Point offset = Point(size,size);
  double minVal;
  double maxVal;
  Point minLoc;
  Point maxLoc;

  for(int y=5; y<=(edgeMap.rows - size); y+=size) {
      for(int x=5; x<=(edgeMap.cols - size); x+=size) {
          current_pt.x = x;
          current_pt.y = y;
          Rect region = Rect(current_pt, current_pt+offset);
          region_frame = edgeMap(region);
          minMaxLoc( region_frame, &minVal, &maxVal, &minLoc, &maxLoc );
          if (isValidSeed(y+maxLoc.y, x+maxLoc.x, edgeMap, gradMap)) {
            dst.push_back(Point(maxLoc.y+y, maxLoc.x+x));
          }
      }
  }
  t = clock() - t;
  std::printf("\tI extracted %d seeds in %f ms\n", static_cast<int>(dst.size()),
              ((float)t)/CLOCKS_PER_SEC*1000.0);
  return static_cast<double> (t) / CLOCKS_PER_SEC*1000.0;
}

void linearFit(const vec_iter_t& start, const vec_iter_t& end,
                std::array<double, 3>& params, double& error)
{
  double sumX, sumY, sumXY, sumX2;
  sumX = sumY = sumXY = sumX2 = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
    sumX += i->x;
    sumY += i->y;
    sumXY += (i->x)*(i->y);
    sumX2 += (i->x)*(i->x);
  }
  size_t lineLength = end - start;
  double xMean = sumX / lineLength;
  double yMean = sumY / lineLength;
  double denom = sumX2 - sumX * xMean;
  if (std::fabs(denom) < 1e-6) {
    //vertical line
    params[0] = 1.0;
    params[1] = 0.0;
    params[2] = -xMean;
  } else {
    params[0] = -(sumXY - sumX * yMean) / denom;
    params[1] = 1.0;
    params[2] = - (yMean + params[0] * xMean);
  }
  double magn = pow(params[0],2.0) + pow(params[1],2.0);
  error = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
    error += pow((params[0]*(i->x) + params[1]*(i->y) + params[2]), 2.0)/magn;
  }
  error = sqrt(error/lineLength);
}

void incLinearFit(const Point2d& pt, std::array<double, 5>& record,
                      std::array<double, 3>& model) {
  record[0] += pt.x;        // sumX
  record[1] += pt.y;        // sumY
  record[2] += pt.x * pt.y; // sumXY
  record[3] += pt.x * pt.x; // sumX2
  record[4] += 1;           // length

  double xMean = record[0] / record[4];
  double yMean = record[1] / record[4];
  double denom = record[3] - record[0] * xMean;

  if (abs(denom) < 1e-6) {
    model[0] = 1.0;
    model[1] = 0.0;
    model[2] = -xMean;
  } else {
    model[0] = - (record[2] - record[0] * yMean) / denom;
    model[1] = 1.0;
    model[2] = - (yMean + model[0] * xMean);
  }
}

double linearFitError( const Point2d& pt, const std::array<double, 3>& model) {
  return pow(model[0] * pt.x + model[1] * pt.y + model[2], 2.0);
}

void circularFit(const vec_iter_t& start, const vec_iter_t& end,
                  std::array<double, 3>& params, double& error) {
  // https://dtcenter.org/met/users/docs/write_ups/circle_fit.pdf

  // find the mean values for x, y
  double x_bar = 0.;
  double y_bar = 0.;
  int N = end - start;
  for (vec_iter_t i = start; i != end; i++) {
    x_bar += i->x;
    y_bar += i->y;
  }
  x_bar /= N;
  y_bar /= N;


  double u, v;
  double Suu, Suv, Svv, Suuu, Svvv, Suvv, Svuu;
  Suu = Suv = Svv = Suuu = Svvv = Suvv = Svuu = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
      u = i->x - x_bar;
      v = i->y - y_bar;
      Suu  += u*u;
      Suv  += u*v;
      Svv  += v*v;
      Suuu += u*u*u;
      Svvv += v*v*v;
      Suvv += u*v*v;
      Svuu += v*u*u;
  }

  //
  // std::printf("x_bar: %f || y_bar: %f || N: %d\n", x_bar, y_bar, N);
  // std::printf("Suu: %f || Suv:%f || Svv:%f\n", Suu, Suv, Svv);
  // std::printf("Suuu: %f || Svvv:%f || Suvv:%f || Svuu:%f\n", Suuu, Svvv, Suvv, Svuu);

  double v_c, u_c, r2;
  u_c = 0.5 * (Suuu+Suvv - Suv/Svv * (Svvv +Svuu)) / (Suu - pow(Suv, 2)/Svv);
  v_c = (0.5 *(Svvv+Svuu) - Suv * u_c )/Svv;
  r2 = pow(u_c, 2) + pow(v_c, 2) + (Suu+Svv)/N;
  params[0] = u_c + x_bar;
  params[1] = v_c + y_bar;
  params[2] = sqrt(r2);

  error = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
      error += pow(pow(i->x - params[0],2) + pow(i->y - params[1],2) - r2 , 2 );
  }
  error = sqrt(error/N);
}

void incSlopeFit(const double& x_n, std::array<double, 5>& record) {
  // http://datagenetics.com/blog/november22017/index.html
  // record is {mu_n, mu_(n-1), S_n, S_(n-1), n}
  // S = sigma^2 * n
  record[4] += 1.;
  record[0] = record[1] + (x_n - record[1])/record[4];
  if (record[4] == 1) {
    record[1] = record[0];
  }
  record[2] = record[3] + (x_n - record[1])*(x_n - record[0]);

  // update old values
  record[1] = record[0];
  record[3] = record[2];
}

inline bool checkSlopeFit(const double& x_n, const std::array<double, 5>& record,
                          const double& threshold) {
  double std_dev = pow(record[2]/record[4], 0.5);
  return abs(x_n - record[0])/std_dev < threshold ;
}

inline double getContourAngle( Mat gradMap[], const Point& pt) {
  const short grad_x = gradMap[0].at<short>(pt.x, pt.y);
  const short grad_y = gradMap[1].at<short>(pt.x, pt.y);
  double ret = atan2(-(float) grad_x, (float)grad_y);
  if (ret < 0.) ret += 2.0 * PI;
  return ret;
}

inline double getAngleDifference( const double theta1, const double theta2) {
  double diff = theta2 - theta1;
  if (diff > PI)  diff = diff - 2. * PI;
  if (diff < -PI) diff = diff + 2. * PI;
  return diff;
}

// bool exploreContour(const Point& seed, Mat& edgeMap, Mat gradMap[],
//                     std::vector<cv::Point>& contour, const int explore_length=4) {
//     // add early failure detection
//     int grad_id;
//     cv::Point new_pt = Point(seed);
//     std::vector<float> v_grad_ids;
//     int edgeVal;
//     int del_grad_id;
//     float tmp;
//     for (int i = 0; i < explore_length; i++) {
//       grad_id = getGradID(gradMap, new_pt);
//       del_grad_id = moveAlongContour(new_pt, grad_id, edgeMap, edgeVal);
//       if (edgeVal < 15) return false;
//       contour.push_back(new_pt);
//       // tmp = grad_id + 0.5*del_grad_id;
//       // if (tmp < 0) tmp += 8;
//       // v_grad_ids.push_back(tmp);
//       v_grad_ids.push_back(getContourAngle(gradMap, new_pt));
//     }
//     std::reverse(std::begin(contour), std::end(contour));
//     std::reverse(std::begin(v_grad_ids), std::end(v_grad_ids));
//     contour.push_back(seed);
//     // v_grad_ids.push_back(getGradID(gradMap,seed));
//     v_grad_ids.push_back(getContourAngle(gradMap, seed));
//
//     new_pt = Point(seed);
//     for (int i = 0; i < explore_length; i++) {
//       grad_id = (getGradID(gradMap, new_pt)+4) % 8;
//       del_grad_id = moveAlongContour(new_pt, grad_id, edgeMap, edgeVal);
//       if (edgeVal < 15) return false;
//       contour.push_back(new_pt);
//       // tmp = grad_id + 4 + 0.5*del_grad_id;
//       // if (tmp > 8) tmp -= 8;
//       // v_grad_ids.push_back(((2*grad_id+del_grad_id+8)%16)/2.);
//       v_grad_ids.push_back(getContourAngle(gradMap, new_pt));
//     }
//
//     cv::Point3f lineParams;
//     double lin_error;
//     linearFit(contour.begin(), contour.end(), lineParams, lin_error);
//     if (lin_error > 1.) return false;
//
//     int g1 = v_grad_ids.front();
//     int g2 = v_grad_ids[explore_length];
//     int g3 = v_grad_ids.back();
//     // if (diff >= 2) std::cout << "\t";
//
//     return subtractGradID_abs(g1, g2) < 2 & subtractGradID_abs(g2, g3) < 2 &
//                 subtractGradID_abs(g1, g3) < 2;
// }

void expandBranch(const Point& seed, Mat& edgeMap, Mat gradMap[], Mat& Seen,
                    std::vector<cv::Point>& contour, std::array<double,3 >& model,
                    const bool direction=true, const int max_length=10)
{
  double alpha = 0.05;
  double angle_new, angle_old, d_angle_old, d_angle_new, dd_angle;
  int edgeVal, grad_id;
  double fit_error = 0.0;
  double tol = 0.05;
  double tol_min = 0.002;
  int alarm = 0;
  std::array<double, 5> record = {{0., 0., 0., 0., 0.}};

  cv::Point new_pt = Point(seed);
  cv::Point2d curv_data = Point2d(0., getContourAngle(gradMap, new_pt));
  incLinearFit(curv_data, record, model);
  d_angle_old = 0.;
  angle_old = getContourAngle(gradMap, seed);
  for (int i = 0; i < max_length; i++) {
    grad_id = getGradID(gradMap, new_pt);
    if (direction) grad_id = (grad_id+4)%8;
    moveAlongContour(new_pt, grad_id, edgeMap);
    edgeVal = edgeMap.at<uchar>(new_pt.x, new_pt.y);

    angle_new = getContourAngle(gradMap, new_pt);
    d_angle_new = getAngleDifference(angle_old, angle_new);

    curv_data.x += 1.;
    curv_data.y = angle_old + d_angle_new;

    dd_angle = d_angle_new - d_angle_old;
    d_angle_old = d_angle_new;
    angle_old = angle_new;

    tol *= 0.85;
    if (tol < tol_min) tol = tol_min;
    if (i > 2) {
      fit_error = linearFitError(curv_data, model);
    }
    if (fit_error < tol) {
      incLinearFit(curv_data, record, model);
    } else {
      alarm += 2;
    }
    if (alarm > 0) alarm -= 1;
    // TERMINATION CONDITIONS
    if (edgeVal < 15) return;
    if (abs(dd_angle) > 0.2) return;
    if (alarm > 5) {
      contour.pop_back();
      contour.pop_back();
      contour.pop_back();
      return;
    }
    contour.push_back(new_pt);
    Seen.at<uchar>(new_pt.x, new_pt.y) = 255;
  }
}

void expandBranch2(const Point& seed, Mat& edgeMap, Mat gradMap[], Mat& Seen,
                    std::vector<cv::Point>& contour, std::array<double,3 >& model,
                    const int max_length=10, bool store_data=false)
{
  std::vector<cv::Point> contour_left, contour_right;
  double angle_new_left, angle_old_left, d_angle_left;
  double angle_new_right, angle_old_right, d_angle_right;
  d_angle_left = d_angle_right = 0.;
  angle_old_left = angle_old_right = getContourAngle(gradMap, seed);
  cv::Point2d curv_data_left = Point2d(0., 0.);
  cv::Point2d curv_data_right = Point2d(0., 0.);

  int edgeVal, grad_id;
  double tol = 0.1;
  double tol_min = 0.02;
  double decay_rate = 0.85;
  double fit_error = 0.0;
  int alarm_left = 0;
  int alarm_right = 0;
  std::array<double, 5> record = {{0., 0., 0., 0., 0.}};

  // left is -1, right is +1
  bool left_alive = true;
  bool right_alive = true;
  cv::Point pt_left(seed);
  cv::Point pt_right(seed);
  std::ofstream myfile;
  if (store_data) {
    myfile.open("../data.txt");
  }


  incLinearFit(curv_data_left, record, model);
  if (store_data) {
    myfile << curv_data_left.x << "," << curv_data_left.y << "\n";
  }
  for (int i = 0; i < max_length/2; i++) {
    tol *= decay_rate;
    // std::printf("Left alive: %d, right_alive: %d\n", left_alive, right_alive);
    if ( !left_alive || !right_alive || (tol < tol_min)) tol = tol_min;

    if (left_alive) {
      grad_id = getGradID(gradMap, pt_left);
      moveAlongContour(pt_left, grad_id, edgeMap);
      edgeVal = edgeMap.at<uchar>(pt_left.x, pt_left.y);
      angle_new_left = getContourAngle(gradMap, pt_left);
      d_angle_left = getAngleDifference(angle_old_left, angle_new_left);
      angle_old_left = angle_new_left;

      curv_data_left.x += 1.;
      curv_data_left.y += d_angle_left;
      if (store_data) {
        myfile << curv_data_left.x << "," << curv_data_left.y << "\n";
      }

      if (i > 3) {
        fit_error = linearFitError(curv_data_left, model);
      } else {
        fit_error = 0.;
      }
      if (fit_error < tol) {
        incLinearFit(curv_data_left, record, model);
      } else {
        alarm_left += 3;
      }
      if (alarm_left > 0) alarm_left -= 1;

      // TERMINATION CONDITIONS
      // std::printf("L: edgeval: %d; alarm: %d\n", edgeVal, alarm_left);
      if ( (edgeVal < 15) || (alarm_left > 5) ) {
        contour_left.erase(contour_left.end()
                            - std::min(3, static_cast<int>(contour_left.size())),
                          contour_left.end());
        left_alive = false;
      } else {
        contour_left.push_back(pt_left);
        Seen.at<uchar>(pt_left.x, pt_left.y) = 255;
      }
    }
    if (right_alive) {
      // shift grad_id to move in opposite direction
      grad_id = (4 + getGradID(gradMap, pt_right)) % 8;
      moveAlongContour(pt_right, grad_id, edgeMap);
      edgeVal = edgeMap.at<uchar>(pt_right.x, pt_right.y);
      angle_new_right = getContourAngle(gradMap, pt_right);
      d_angle_right = getAngleDifference(angle_old_right, angle_new_right);
      angle_old_right = angle_new_right;

      curv_data_right.x -= 1.;
      curv_data_right.y += d_angle_right;
      if (store_data) {
        myfile << curv_data_right.x << "," << curv_data_right.y << "\n";
      }
      if (i > 3) {
        fit_error = linearFitError(curv_data_right, model);
      } else {
        fit_error = 0.;
      }
      if (fit_error < tol) {
        incLinearFit(curv_data_right, record, model);
      } else {
        alarm_right += 3;
      }
      if (alarm_right > 0) alarm_right -= 1;

      // TERMINATION CONDITIONS
      // std::printf("R: edgeval: %d; alarm: %d\n", edgeVal, alarm_right );
      if ( (edgeVal < 15) || (alarm_right > 5) ) {
        contour_right.erase(contour_right.end()
                              - std::min(3, static_cast<int>(contour_right.size())),
                          contour_right.end());
        right_alive = false;
      } else {
        contour_right.push_back(pt_right);
        Seen.at<uchar>(pt_right.x, pt_right.y) = 255;
      }
    }
    if ( !(left_alive or right_alive)) {
      break;
    }
  } // end for

  // now we can assemble contour from left and right segments
  // std::printf("left size: %d || right size: %d\n",
  //             static_cast<int>(contour_left.size()),
  //             static_cast<int>(contour_right.size()));
  contour.reserve(contour_left.size()+contour_right.size()+1);
  copy(contour_left.rbegin(),contour_left.rend(),back_inserter(contour));
  contour.push_back(seed);
  copy(contour_right.begin(),contour_right.end(),back_inserter(contour));
  if (store_data)  myfile.close();
} // end function

void closestPoint_line(const std::array<double,3>& model, const Point& PoI,
                          Point2d& closest_pt) {
  Point2d some_pt;
  Point2d n(model[0],model[1]);
  if (model[0] == 0.0) {
    //vertical line
    some_pt.x = PoI.x;
    some_pt.y = -model[2];
  } else {
    if (abs(model[0]) < 1.0) {
      n.x = model[0]; n.y = model[1];
      some_pt.x = PoI.x;
      some_pt.y = - ((float) model[0]*some_pt.x + model[2]) / model[1];
    } else {
      n.x = model[1]; n.y = model[0];
      some_pt.y = PoI.y;
      some_pt.x = - ((float) model[1]*some_pt.y + model[2]) / model[0];
    }
  }
  double n_magn = pow( (pow(n.x, 2.) + pow(n.y, 2.) ), 0.5 );
  Point2d vec(PoI.x - some_pt.x, PoI.y - some_pt.y);
  double n_1 = (vec.x * n.x + vec.y * n.y) / n_magn;
  closest_pt.x = PoI.x - n_1 * n.x; closest_pt.y = PoI.y - n_1 * n.y;
  // std::printf("\n" );
  // std::printf("some_pt (%f, %f)\n", some_pt.x, some_pt.y);
  // std::printf("PoI (%d, %d)\n", PoI.x, PoI.y);
  // std::printf("n (%f, %f)\n", n.x, n.y);
  // std::printf("vec (%f, %f)\n", vec.x, vec.y);
  // std::printf("n_1: %f\n", n_1);
  // std::printf("closest_pt (%f, %f)\n", closest_pt.x, closest_pt.y);
  // std::printf("\n" );

}

void localizeContour(const std::vector<Point>& contour, int& contour_type,
                     std::array<double, 5>& location) {
  // location = {x1, y1, x2, y2, empty} for a line
  // location = {cx, cy, theta1, theta2, radius} for an arc

  int contour_length = static_cast<int> (contour.size());
  int num_pts_sampled = 5;
  std::vector<cv::Point> sampled_pts;
  std::array<double, 3> model;
  double error = 0.0;

  for (int attempts = 0; attempts != 2; attempts++) {
    sampled_pts.clear();
    for (int i= 0; i < num_pts_sampled; i++) {
      sampled_pts.push_back( Point(contour[ rand() % contour_length]) );
    }
    linearFit(sampled_pts.begin(), sampled_pts.end(), model, error);
    if (error < 2) {
      linearFit(contour.begin(), contour.end(), model, error);
      if (error < 1) {
        Point2d closest_pt;
        closestPoint_line(model, contour.front(), closest_pt);
        location[0] = closest_pt.x;        // x1
        location[1] = closest_pt.y;        // y1
        closestPoint_line(model, contour.back(), closest_pt);
        location[2] = closest_pt.x;        // x2
        location[3] = closest_pt.y;        // y2
        location[4] = -model[0]/model[1];   // slope
        contour_type = 0;
        return;
      }
    }
  }
  // attempt a circle
  circularFit(contour.begin(), contour.end(), model, error);
  // if (error > 2) {
  //   contour_type = -1;
  //   return;
  // }
  location[0] = model[0];                             // cx
  location[1] = model[1];
  Point2d v1(contour.front().x - model[0],
                  contour.front().y - model[1]);
  Point2d v2(contour.back().x - model[0],
                  contour.back().y - model[1]);
  Point2d v_C(contour[contour_length/2].x - model[0],
                  contour[contour_length/2].y - model[1]);
  Point2d v_m(contour[5].x - contour.front().x,
            contour[5].y - contour.front().y);
  double curl = v1.x*v_m.y - v1.y*v_m.x;
  double mid_angle = atan2(v_C.x, v_C.y);

  double angle_diff1 = acos( (v1.x*v_C.x + v1.y*v_C.y) /
                          (pow(pow(v1.x,2) +pow(v1.y,2), 0.5) *
                           pow(pow(v_C.x,2) +pow(v_C.y,2), 0.5) ) );
 double angle_diff2 = acos( (v_C.x*v2.x + v_C.y*v2.y) /
                         (pow(pow(v_C.x,2) +pow(v_C.y,2), 0.5) *
                          pow(pow(v2.x,2) +pow(v2.y,2), 0.5) ) );
  location[2] = mid_angle - curl/abs(curl) * angle_diff1;   // theta1
  location[3] = mid_angle + curl/abs(curl) * angle_diff2;    // theta2

  location[4] = model[2];                             // radius
  contour_type = 1; // circle fit good
}

void extractContours(Mat & img_gray, Mat& contour_map) {
  Mat Seen = Mat::zeros(img_gray.size(), CV_8U);

  bool time_it = true;
  float t_total = 0.0;
  std::printf("--- Running Contour Detection (%d x %d pixels) ---\n\n",
              img_gray.cols, img_gray.rows );
  t_total += suppressNoise(img_gray, img_gray, time_it);

  Mat edgeMap;
  Mat gradMap[] = {Mat(img_gray.rows, img_gray.cols, CV_16S),
                   Mat(img_gray.rows, img_gray.cols, CV_16S)};
  t_total += computeEdgeAndGradMap(img_gray, edgeMap, gradMap, time_it);
  std::vector<cv::Point> seeds;
  t_total += extractSeeds(edgeMap, gradMap, seeds, time_it);

  cv::cvtColor(edgeMap, contour_map, CV_GRAY2BGR);

  // interesting points for testing
  Point circle[] = {Point(104,197)};
  Point ellipse[] = {Point(83,331)};
  Point smaller_circle[] = {Point(213,332)};
  Point toyblocks[] = {Point(291,36), Point(231,21), Point(82,207),
                          Point(319,155),Point(304,275),Point(314,333),
                          Point(188,118)};
  Point occlusion2[] = {Point(269,468), Point(329,127), Point(128,154),
                          Point(126,364), Point(182,228), Point(39, 443),
                          Point(151,330), Point(393,121)};
  Point line_edges[] = {Point(262,133), Point(246,300), Point(298, 482)};
  // seeds.clear();
  // for (const auto& pt: toyblocks) {
  //   seeds.push_back(pt);
  // }
  // seeds.push_back(Point(66,349));

  clock_t t = clock();
  std::vector<cv::Point> contour;
  std::array<double, 3> model;
  int counter = 950;
  Point seed;
  for (int i=0; i < seeds.size(); i+=1) {
    seed = seeds[i];
    //

    if (!shiftSeed(seed, edgeMap, gradMap)) {
      // std::printf("Failed to shift seed\n");
      continue;
    }
    if (Seen.at<uchar>(seed.x, seed.y) != 0) {
      // std::printf("Already explored this point\n");
      continue;
    }
    if (!isStableSeed(seed, edgeMap, gradMap, 2)) {
      // std::printf("Unstable seed\n");
      continue;
    }

    int length = 600;
    contour.clear();
    expandBranch2(seed, edgeMap, gradMap, Seen, contour, model, length);

    if (contour.size() < 9) {
      // std::printf("Contour too short: length %d\n", static_cast<int>(contour.size()));
      continue;
    }
    counter++;

    int contour_type = -1;
    std::array<double,5> location;
    double est_radius = abs(model[1]/model[0]);
    if (est_radius > std::max(img_gray.cols, img_gray.rows)) {
      // now we have to fit it
      std::array<double, 3> model;
      int offset = 0;
      double fit_error;
      linearFit(contour.begin()+offset, contour.end()-offset,
                model, fit_error);
      if (fit_error < 2) {
        // continue;
        contour_type = 0;
        Point2d closest_pt;
        closestPoint_line(model, contour.front(), closest_pt);
        location[0] = closest_pt.x;        // x1
        location[1] = closest_pt.y;         // y1
        closestPoint_line(model, contour.back(), closest_pt);
        location[2] = closest_pt.x;        // x2
        location[3] = closest_pt.y;        // y2
        location[4] = -model[0]/model[1];   // slope
      }
    }
    if (contour_type == -1) {
      // continue;
      localizeContour(contour, contour_type, location);
    }
    if (contour_type == 0) {
        // std::printf("Discovered a line\n" );
        Point pt1(round(location[1]),round(location[0]));
        Point pt2(round(location[3]),round(location[2]));
        // std::printf("(%d, %d) to (%d, %d)\n\n", pt1.x, pt1.y, pt2.x, pt2.y);
        cv::line(contour_map, pt1, pt2, Scalar(0,150,255),1);
    } else if (contour_type == 1) {
      // std::printf("Discovered an arc\n" );
      Point center(location[1],location[0]);
      int radius = static_cast<int>(round(location[4]));
      // std::printf("centered at (%d, %d), r = %d\n\n", center.x, center.y, radius);
      double startAngle = 180*location[2]/PI;
      double endAngle = 180*location[3]/PI;
      // std::printf("start: %f || end: %f \n",startAngle, endAngle );
      // cv::circle(contour_map, center, radius, Scalar(0,255,0));
      cv::ellipse(contour_map, center, cv::Size(radius, radius), 0.0,
                  startAngle, endAngle, Scalar(180,0,255), 1);
    }

    // contour_map all points in a contour
    // int j=0;
    // for (auto& pt: contour) {
    //   contour_map.at<Vec3b>(pt.x,pt.y)[0] = 0;
    //   contour_map.at<Vec3b>(pt.x,pt.y)[1] = 0;
    //   contour_map.at<Vec3b>(pt.x,pt.y)[2] = 255;
    //   j++;
    // }
    // // // // // //
    // contour_map.at<Vec3b>(seed.x,seed.y)[0] = 255;
    // contour_map.at<Vec3b>(seed.x,seed.y)[1] = 0;
    // contour_map.at<Vec3b>(seed.x,seed.y)[2] = 0;

  }

  t = clock() - t;
  t_total += ((float)t)/CLOCKS_PER_SEC*1000.0;
  std::printf("\tI explored %d contours in %f ms\n", counter,
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  std::printf("\n--- TOTAL: %f ms ---\n", t_total);

  // Mat m_grad_id = Mat(img_gray.rows, img_gray.cols, CV_8U);
  // for (int i=0; i != m_grad_id.cols; i++ ) {
  //   for (int j=0; j != m_grad_id.rows; j++) {
  //     m_grad_id.at<uchar>(j,i) = 40*getGradID(gradMap, cv::Point(j,i));
  //   }
  // }
  // imwrite( "../Results/Contours.png", color );
  // namedWindow("Seed Map", WINDOW_NORMAL );
  // moveWindow("Seed Map", 0,30);
  // resizeWindow("Seed Map", 800,600);
  // imshow("Seed Map", contour_map );
  // waitKey(0);
}


#endif
