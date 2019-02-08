#ifndef CONTOURDETECTION_INCLUDE
#define CONTOURDETECTION_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include <algorithm>
#include <math.h>
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

inline int subtractGradID(const int& g1, const int& g2) {
  std::printf("subtractGradID not working\n" );
  int diff = abs(g1 - g2);
  if (diff > 4) diff = 8 - diff;
  return diff;
}

void computeEdgeAndGradMap(Mat& image_gray, Mat& edgeMap, Mat gradMap[]) {
  Mat abs_grad_x, abs_grad_y;
  int ddepth = CV_16S;
  cv::Sobel( image_gray, gradMap[0], ddepth, 1, 0, 3 );
  cv::Sobel( image_gray, gradMap[1], ddepth, 0, 1, 3 );
  cv::convertScaleAbs( gradMap[0], abs_grad_x, 0.3);
  cv::convertScaleAbs( gradMap[1], abs_grad_y, 0.3);
  cv::add(abs_grad_x, abs_grad_y, edgeMap);
}

inline int moveAlongContour(cv::Point& new_pt, const int& grad_id,
                              const Mat& edgeMap, int& edgeVal) {
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

    edgeVal = static_cast<int>(edgeMap.at<uchar>(new_pt.x, new_pt.y));
    return ret;
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

void extractSeeds(Mat& edgeMap, Mat gradMap[], std::vector<Point>& dst, int size=15, int threshold = 15) {
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
}

void linearFit(const vec_iter_t& start, const vec_iter_t& end,
                cv::Point3f& params, double& error)
{
  double sumX, sumY, sumXY, sumX2;
  sumX = sumY = sumXY = sumX2 = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
    sumX += i->x;
    sumY += i->y;
    sumXY += i->x*i->y;
    sumX2 += i->x*i->x;
  }
  size_t lineLength = end - start;
  double xMean = sumX / lineLength;
  double yMean = sumY / lineLength;
  double denom = sumX2 - sumX * xMean;
  double A, B, C;
  if (std::fabs(denom) < 1e-7) {
    //vertical line
    A = 1.0;
    B = 0.0;
    C = -xMean;
  } else {
    A = -(sumXY - sumX * yMean) / denom;
    B = 1.0;
    C = - (yMean + A * xMean);
  }

  params.x = A; params.y = B; params.z = C;
  params /= sqrt(pow(A,2)+B);

  error = 0.0;
  for (vec_iter_t i = start; i != end; i++) {
    error += pow(A*(i->x) + B*(i->y) +C, 2.0);
  }
  error = sqrt(error/lineLength);
}

// void quadraticFit(const vec_iter_t& start, const vec_iter_t& end,
//                   cv::Point3f& params, double& error)
// {
  //
//   int n_points = static_cast<int> (end-start);
//   MatrixXd X(n_points,3);
//   VectorXd Y(n_points);
//   VectorXd B(3);
//   int counter = 0;
//   for (vec_iter_t i = start; i != end; i++, counter++) {
//     X(counter, 0) = 1.0;
//     X(counter, 1) = i->x;
//     X(counter, 2) = pow(i->x , 2.0);
//     Y(counter) = i->y;
//   }
//   MatrixXd X_transpose = X.transpose();
//   MatrixXd X_squared = X_transpose*X;
//   if (abs(X_squared.determinant()) < 0.00001) {
//     // this will signify that the fit was bad
//     error = 1e6;
//     return;
//   }
//   B = X_squared.inverse() * X_transpose * Y;
//   params.x = B[2]; params.y = B[1]; params.z = B[0];
//   error = 0.0;
//   for (vec_iter_t i = start; i != end; i++) {
//     error += pow( B[0] + B[1]*(i->x) + B[2] * pow(i->x,2.0) - i->y  , 2.0);
//   }
// }

void incLinearFit(const Point2d& pt, std::array<double, 4>& record,
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

bool exploreContour(const Point& seed, Mat& edgeMap, Mat gradMap[],
                    std::vector<cv::Point>& contour, const int explore_length=4) {
    // add early failure detection
    int grad_id;
    cv::Point new_pt = Point(seed);
    std::vector<float> v_grad_ids;
    int edgeVal;
    int del_grad_id;
    float tmp;
    for (int i = 0; i < explore_length; i++) {
      grad_id = getGradID(gradMap, new_pt);
      del_grad_id = moveAlongContour(new_pt, grad_id, edgeMap, edgeVal);
      if (edgeVal < 15) return false;
      contour.push_back(new_pt);
      // tmp = grad_id + 0.5*del_grad_id;
      // if (tmp < 0) tmp += 8;
      // v_grad_ids.push_back(tmp);
      v_grad_ids.push_back(getContourAngle(gradMap, new_pt));
    }
    std::reverse(std::begin(contour), std::end(contour));
    std::reverse(std::begin(v_grad_ids), std::end(v_grad_ids));
    contour.push_back(seed);
    // v_grad_ids.push_back(getGradID(gradMap,seed));
    v_grad_ids.push_back(getContourAngle(gradMap, seed));

    new_pt = Point(seed);
    for (int i = 0; i < explore_length; i++) {
      grad_id = (getGradID(gradMap, new_pt)+4) % 8;
      del_grad_id = moveAlongContour(new_pt, grad_id, edgeMap, edgeVal);
      if (edgeVal < 15) return false;
      contour.push_back(new_pt);
      // tmp = grad_id + 4 + 0.5*del_grad_id;
      // if (tmp > 8) tmp -= 8;
      // v_grad_ids.push_back(((2*grad_id+del_grad_id+8)%16)/2.);
      v_grad_ids.push_back(getContourAngle(gradMap, new_pt));
    }

    cv::Point3f lineParams;
    double lin_error;
    linearFit(contour.begin(), contour.end(), lineParams, lin_error);
    if (lin_error > 1.) return false;

    int g1 = v_grad_ids.front();
    int g2 = v_grad_ids[explore_length];
    int g3 = v_grad_ids.back();
    // if (diff >= 2) std::cout << "\t";

    return subtractGradID_abs(g1, g2) < 2 & subtractGradID_abs(g2, g3) < 2 &
                subtractGradID_abs(g1, g3) < 2;
}

void expandBranch(const Point& seed, Mat& edgeMap, Mat gradMap[],
                    std::vector<cv::Point>& contour, const bool direction=true,
                    const int max_length=10)
{
  double alpha = 0.05;
  double angle_new, angle_old, d_angle_old, d_angle_new, dd_angle;
  int edgeVal, grad_id;
  double fit_error = 0.0;
  if (abs(dd_angle) > 0.2) return;
  double tol = 0.05;
  double tol_min = 0.002;
  int alarm = 0;
  std::array<double, 4> record = {{0., 0., 0., 0.}};
  std::array<double, 3> model = {{0., 0., 0.0}};


  cv::Point new_pt = Point(seed);
  cv::Point2d curv_data = {0., getContourAngle(gradMap, new_pt)};
  incLinearFit(curv_data, record, model);
  d_angle_old = 0.;
  angle_old = getContourAngle(gradMap, seed);
  for (int i = 0; i < max_length; i++) {
    grad_id = getGradID(gradMap, new_pt);
    if (direction) grad_id = (grad_id+4)%8;
    moveAlongContour(new_pt, grad_id, edgeMap, edgeVal);

    // TERMINATION CONDITIONS
    if (edgeVal < 15) return;
    if (abs(dd_angle) > 0.2) return;

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
      // std::printf("OUTLIER DETECTED\n" );
      alarm += 2;
    }
    if (alarm > 0) alarm -= 1;
    if (alarm > 5) {
      contour.pop_back();
      contour.pop_back();
      contour.pop_back();
      return;
    }
    contour.push_back(new_pt);
  }
}

void expandSeed(const Point& seed, Mat& edgeMap, Mat gradMap[]) {
  // Point shifted_seed = shiftSeed(seed, edgeMap, gradMap);
  // if (!exploreContour(shifted_seed, edgeMap, gradMap, contour)) {
  //   return;
  // }
  // extendLine()
  // success = exploreContour(explore_length=7)
  // if success
  //   contour_type = characterizeContour
  //   if contour_type = line:
  //     do (extendLine)
  //       until error > threshold
  //   else if contour_type = circle:
  //     do (extendCircle)
  //       until error > threshold

}

void extractContours(Mat & img_gray) {
  float t_total = 0.0;
  std::printf("--- Running Contour Detection (%d x %d pixels) ---\n\n",
              img_gray.cols, img_gray.rows );
  clock_t t = clock();
  suppressNoise(img_gray, img_gray);
  t = clock()-t; t_total += ((float)t)/CLOCKS_PER_SEC*1000.0;
  std::printf("\tI applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  t = clock();
  Mat edgeMap;
  Mat gradMap[] = {Mat(img_gray.rows, img_gray.cols, CV_16S),
                   Mat(img_gray.rows, img_gray.cols, CV_16S)};
  computeEdgeAndGradMap(img_gray, edgeMap, gradMap);
  t = clock()-t; t_total += ((float)t)/CLOCKS_PER_SEC*1000.0;
  std::printf("\tI computed edge and gradient maps in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  std::vector<cv::Point> seeds;
  t = clock();
  extractSeeds(edgeMap, gradMap, seeds);
  t = clock() - t; t_total += ((float)t)/CLOCKS_PER_SEC*1000.0;
  std::printf("\tI extracted %d seeds in %f ms\n", static_cast<int>(seeds.size()),
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  // visualization for debugging
  Mat color;
  cv::cvtColor(edgeMap, color, cv::COLOR_GRAY2BGR);

  Point circle[] = {{74,244}};
  Point ellipse[] = {{83,331}};
  Point smaller_circle[] = {{213,332}};
  Point toyblocks[] = {{291,36}, {231,21}, {82,207}, {319,155},{304,275},{314,333}};
  Point occlusion2[] = {{269,468}, {329,127}, {128,154}, {126,364}, {182,228},
                        {151,330},{393,121}};
  // seeds.clear();
  // for (const auto& pt: toyblocks) {
  //   seeds.push_back(pt);
  // }
  // seeds.push_back(toyblocks[2]);

  t = clock();
  std::vector<cv::Point> contour;
  int counter = 0;
  for (auto& seed: seeds) {
    contour.clear();
    // std::printf("Seed at %d, %d with G: %d\n", seed.x, seed.y, getGradID(gradMap, seed));
    // color.at<Vec3b>(seed.x,seed.y)[0] = 0;
    // color.at<Vec3b>(seed.x,seed.y)[1] = 255;
    // color.at<Vec3b>(seed.x,seed.y)[2] = 255;
    if (!shiftSeed(seed, edgeMap, gradMap)) continue;
    if (!exploreContour(seed, edgeMap,gradMap, contour, 2)) {
      continue;
    }


    int length = 500;
    contour.clear();
    expandBranch(seed, edgeMap, gradMap, contour, true, length/2);
    std::reverse(std::begin(contour), std::end(contour));
    contour.push_back(seed);

    // std::ofstream myfile;
    // myfile.open("../data.txt");
    // myfile << contour.size()-1 << ", ";

    expandBranch(seed, edgeMap, gradMap, contour, false, length/2);
    if (contour.size() < 9) continue;
    counter++;

    int i = 0;
    for (auto& pt: contour) {
      color.at<Vec3b>(pt.x,pt.y)[0] = 0;
      color.at<Vec3b>(pt.x,pt.y)[1] = 0;
      color.at<Vec3b>(pt.x,pt.y)[2] = 255;
      // myfile << getContourAngle(gradMap, pt);
      // if (i != contour.size()-1) myfile << ", ";
      i++;
    }
    // myfile.close();

    color.at<Vec3b>(seed.x,seed.y)[0] = 255;
    color.at<Vec3b>(seed.x,seed.y)[1] = 0;
    color.at<Vec3b>(seed.x,seed.y)[2] = 0;
    color.at<Vec3b>(contour.front().x,contour.front().y)[0] = 0;
    color.at<Vec3b>(contour.front().x,contour.front().y)[1] = 255;
    color.at<Vec3b>(contour.front().x,contour.front().y)[2] = 0;
    color.at<Vec3b>(contour.back().x,contour.back().y)[0] = 0;
    color.at<Vec3b>(contour.back().x,contour.back().y)[1] = 255;
    color.at<Vec3b>(contour.back().x,contour.back().y)[2] = 0;
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
  namedWindow("Seed Map", WINDOW_NORMAL );
  moveWindow("Seed Map", 0,30);
  resizeWindow("Seed Map", 800, 650);
  imshow("Seed Map", color );
  waitKey(0);
}


#endif
