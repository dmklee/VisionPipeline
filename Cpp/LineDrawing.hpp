#ifndef LINE_DRAWING_INCLUDE
#define LINE_DRAWING_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "ContourDetection.hpp"

struct Line {
  // _A * x + _B * y + _C = 0
  float _A,_B,_C;
  double _sumX, _sumY, _sumXY, _sumX2;
  std::vector<cv::Point> _data; // may or may not be needed
};

typedef std::vector< Line > lineChain_type;
typedef std::vector< lineChain_type > lineChainList_type;
typedef seg_type::iterator seg_it_type;
typedef std::array<cv::Point,2> ray_t;


inline bool stabilizeAnchor(cv::Point& pt, const Mat& edgeMap,
                            const Mat gradMap[],
                            const int gradThreshold,
                            const int peakThreshold) {
  int NUM_ATTEMPTS = 5;
  uchar val;
  int i;
  bool done = false;
  int grad_id;
  for (i=0; i < NUM_ATTEMPTS; i++) {
    val = edgeMap.at<uchar>( pt.x, pt.y);
    if (val < gradThreshold) return false;
    // walk up contour
    grad_id = getGradID(gradMap, pt) % 4;
    switch (grad_id) {
      case 0:
        if (val < edgeMap.at<uchar>(pt.x, pt.y+1)) {
          pt.y++;
        } else if (val < edgeMap.at<uchar>(pt.x, pt.y-1)) {
          pt.y--;
        } else {
          done = true;
        }
        break;
      case 1:
        if (val < edgeMap.at<uchar>(pt.x-1, pt.y+1)) {
          pt.x--; pt.y++;
        } else if (val < edgeMap.at<uchar>(pt.x+1, pt.y-1)) {
          pt.x++; pt.y--;
        } else {
          done = true;
        }
        break;
      case 2:
        if (val < edgeMap.at<uchar>(pt.x+1, pt.y)) {
          pt.x++;
        } else if (val < edgeMap.at<uchar>(pt.x-1, pt.y)) {
          pt.x--;
        } else {
          done = true;
        }
        break;
      case 3:
        if (val < edgeMap.at<uchar>(pt.x+1, pt.y+1)) {
          pt.x++; pt.y++;
        } else if (val < edgeMap.at<uchar>(pt.x- 1, pt.y-1)) {
          pt.x--; pt.y--;
        } else {
          done = true;
        }
        break;
    }
    if (done) break;
  }
  // return i < NUM_ATTEMPTS-1;
  val = edgeMap.at<uchar>( pt.x, pt.y);
  grad_id = getGradID(gradMap, pt) % 4;
  switch (grad_id) {
    case 0:
      if (val > peakThreshold + edgeMap.at<uchar>(pt.x, pt.y+1) &&
          val > peakThreshold + edgeMap.at<uchar>(pt.x, pt.y-1)) {
        return true;
      }
      break;
    case 1:
      if (val > peakThreshold + edgeMap.at<uchar>(pt.x-1, pt.y+1) &&
          val > peakThreshold + edgeMap.at<uchar>(pt.x+1, pt.y-1)) {
        return true;
      }
      break;
    case 2:
      if (val > peakThreshold + edgeMap.at<uchar>(pt.x+1, pt.y) &&
          val > peakThreshold + edgeMap.at<uchar>(pt.x-1, pt.y)) {
        return true;
      }
      break;
    case 3:
      if (val > peakThreshold + edgeMap.at<uchar>(pt.x-1, pt.y-1) &&
          val > peakThreshold + edgeMap.at<uchar>(pt.x+1, pt.y+1)) {
        return true;
      }
      break;
  }
  return false;
}

double extractAnchors_smart(const Mat& edgeMap, const Mat gradMap[],
                  std::vector<Point>& dst,
                  bool time_it=true, int size=9,
                  const int gradThreshold=15,
                  const int peakThreshold=4) {
  clock_t t = clock();
  Mat region_frame;
  Point current_pt;
  Point anchor;
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
          anchor = current_pt+maxLoc;
          if (stabilizeAnchor(anchor, edgeMap, gradMap, gradThreshold,
                              peakThreshold)) {
            dst.push_back(anchor);
          }
      }
  }
  t = clock() - t;
  if (time_it) {
    std::printf("\tI extracted %d anchors in %f ms\n", static_cast<int>(dst.size()),
                ((float)t)/CLOCKS_PER_SEC*1000.0);
  }
  return static_cast<double> (t) / CLOCKS_PER_SEC*1000.0;
}

void extractEdgeSegment(const Point& anchor, const Mat& edgeMap, const Mat gradMap[],
                        Mat& Seen, std::vector<Point>& edgeSegment,
                        const int gradThreshold=15 )
{
  bool alive_left = true;
  bool alive_right = true;
  bool direction = true;
  int grad_id;
  Point pt;
  for (int i=0; i < 2; i++) {
    pt = Point(anchor);
    do {
      Seen.at<uchar>(pt.x,pt.y) = 255;
      edgeSegment.push_back(pt);
      grad_id = getGradID(gradMap, pt);
      if (direction) grad_id = (grad_id+4)%8;
      moveAlongContour(pt, grad_id, edgeMap);
    } while (edgeMap.at<uchar>(pt.x,pt.y) > gradThreshold &&
            Seen.at<uchar>(pt.x,pt.y) == 0);
    direction = !direction;
  }
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

void leastSquaresLineFit(const std::vector<cv::Point>::iterator& it,
                          const int minLineLength,
                          Line& L, double& error) {

  int x = (*(it+minLineLength)).x;
  int y = (*(it+minLineLength)).y;
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
    error += pow(L._A*(*(it+i)).x + L._B*(*(it+i)).y +L._C, 2.0);
  }
  error = sqrt(error/minLineLength);

  x = (*it).x;
  y = (*it).y;
  L._sumX  -= x;
  L._sumY  -= y;
  L._sumXY -= x*y;
  L._sumX2 -= x*x;
}

inline double computePointDistance2Line(const Line& L,
                              const std::vector<cv::Point>::iterator& it) {
  return abs(L._A * (*it).x + L._B * (*it).y + L._C)/
          std::sqrt(pow(L._A,2)+pow(L._B,2));
}

void computeNearestPoint(const Line& L, const cv::Point& p0,
                               cv::Point nearest) {
  Point2d n(L._A, L._B);
  Point2d p1(-L._C/L._A, 0);
  Point2d u(p1.x-p0.x, p1.y-p0.y);
  double u_1 = (u.x*n.x + u.y*n.y)/pow(pow(n.x,2) + pow(n.y,2), 0.5);
  nearest.x = round(p0.x - u_1*n.x);
  nearest.y = round(p0.y - u_1*n.y);
}

void lineFit(std::vector<cv::Point>::iterator& it, int numPixels,
             lineChain_type& lineChain,
             const int minLineLength, const float maxFitError) {
  // return true unless out of pixels
  double error = 10*maxFitError;
  Line L;
  L._sumX = L._sumY = L._sumXY = L._sumX2 = 0.0;
  int x,y;
  for (int i=0; i<(minLineLength-1); i++) {
    x = (*(it+i)).x;
    y = (*(it+i)).y;
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
    cv::Point tmp( *(it+i) );
    L._data.push_back(tmp);
  }

  int lineLength = minLineLength;
  it += lineLength;
  double d;
  while (lineLength < numPixels) {
    d = computePointDistance2Line(L, it);
    if (d > maxFitError) break;
    cv::Point tmp( *it );
    L._data.push_back(tmp);
    lineLength++;
    it++;
  }
  lineChain.push_back(L);
}

void generateLineChain(std::vector<Point>& edgeSegment, lineChain_type& lineChain,
                    const int minLineLength=12, const float maxFitError=2) {
  std::vector<cv::Point>::iterator it = edgeSegment.begin();
  int segLength = edgeSegment.end()-it;
  while (segLength > minLineLength) {
    lineFit(it, segLength, lineChain, minLineLength, maxFitError);
    segLength = edgeSegment.end()-it;
  }
}

double findLines(const std::vector<cv::Point>& anchors, const Mat& edgeMap,
              const Mat gradMap[], Mat& Seen,
              std::vector<ray_t>& rayChain, bool time_it) {
  // PARAMS
  double MSE_TOL = 1.5;
  double BEND_TOL = 22.5*3.14159265 / 180.;

  clock_t t = clock();
  int minLineLength = computeMinLineLength(edgeMap);
  std::vector<Point> edgeSegment;
  ray_t ray;
  std::vector<Line> lineChain;
  double old_tangent, new_tangent;
  for (const auto& anchor: anchors) {
    edgeSegment.clear();
    extractEdgeSegment(anchor, edgeMap, gradMap, Seen, edgeSegment);
    // if (edgeSegment.size() <= minLineLength) {
    //   continue;
    // }
    generateLineChain(edgeSegment, lineChain, minLineLength, MSE_TOL);
    for (int i=0; i != lineChain.size(); i++) {
      if (i == 0) {
        ray[0] = Point(lineChain[i]._data[3]);
        ray[1] = Point(lineChain[i]._data[lineChain[i]._data.size()-3]);
        old_tangent = atan2(lineChain[i]._A, lineChain[i]._B);
        continue;
      }
      new_tangent = atan2(lineChain[i]._A, lineChain[i]._B);
      if (abs(getAngleDifference(old_tangent, new_tangent)) > BEND_TOL) {
        rayChain.push_back(ray);
        ray[0] = Point(lineChain[i]._data[3]);
      }
      ray[1] = Point(lineChain[i]._data[lineChain[i]._data.size()-3]);
      old_tangent = new_tangent;
    }
    rayChain.push_back(ray);

  }
  t = clock() - t;
  if (time_it) {
    std::printf("\tI found %d line segments in %f ms\n", \
                static_cast<int>(lineChain.size()),
                ((float)t)/CLOCKS_PER_SEC*1000.0);
  }
  return static_cast<double> (t) / CLOCKS_PER_SEC*1000.0;

}

void runLineDrawing(Mat& img, Mat& contourMap) {
  std::printf("--- Running Line Drawing (%d x %d pixels) ---\n\n",
              img.cols, img.rows );
  double t_total = 0.0;
  bool time_it = true;

  t_total += suppressNoise(img,img, time_it);

  Mat edgeMap;
  Mat gradMap[] = {Mat(img.rows, img.cols, CV_16S),
                   Mat(img.rows, img.cols, CV_16S)};
  t_total += computeEdgeAndGradMap(img, edgeMap, gradMap, time_it);

  cv::cvtColor(edgeMap, contourMap, CV_GRAY2BGR);
  Mat Seen = Mat::zeros(img.size(), CV_8U);

  std::vector< Point> anchors;
  t_total += extractAnchors_smart(edgeMap, gradMap, anchors);

  // for (const auto& anchor: anchors) {
  //   contourMap.at<Vec3b>(anchor.x, anchor.y)[0] = 255;
  //   contourMap.at<Vec3b>(anchor.x, anchor.y)[1] = 0;
  //   contourMap.at<Vec3b>(anchor.x, anchor.y)[2] = 0;
  // }

  std::vector<ray_t> rayChain;
  t_total += findLines(anchors, edgeMap, gradMap, Seen, rayChain, time_it);
  for (const auto& ray: rayChain) {
    Point pt1(ray[0].y, ray[0].x);
    Point pt2(ray[1].y, ray[1].x);
    cv::line(contourMap, pt1, pt2,
            cv::Scalar(0,0,255), 1);
  }

  std::printf("\n--- TOTAL: %f ms ---\n", t_total);

}

#endif
