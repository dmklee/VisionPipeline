#ifndef CONTOURDETECTION_INCLUDE
#define CONTOURDETECTION_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include <algorithm>

using namespace cv;

typedef std::vector<cv::Point>::const_iterator vec_iter_t;

// This describes the line, not the gradient
// and in terms of the user viewing the image
enum GRAD_ID {HORIZONTAL =0, UPHILL=1, VERTICAL=2, DOWNHILL=3};

inline GRAD_ID getGradID(const short grad_x, const short grad_y) {
  float ratio = grad_y != 0 ? ((float)grad_x)/grad_y : ((float)grad_x)/0.0001;
  if (abs(ratio) < 0.41) {
    return HORIZONTAL;
  }
  if (ratio > 0.41 and ratio < 2.41) {
    return UPHILL;
  }
  if (ratio > 2.41 or ratio < -2.41) {
    return VERTICAL;
  }
  return DOWNHILL;
}

inline GRAD_ID getGradID(const Mat gradMap[], const Point& pt) {
  const short grad_x = gradMap[0].at<short>(pt.x, pt.y);
  const short grad_y = gradMap[1].at<short>(pt.x, pt.y);
  return getGradID(grad_x, grad_y);
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

inline int moveAlongContour(cv::Point& new_pt, const GRAD_ID& grad_id,
                              const Mat& edgeMap, const bool dir = true) {
    static cv::Point offsets[] = {Point(0,1), Point(-1,1), Point(-1,0),
                                  Point(-1,-1), Point(0,-1), Point(1,-1),
                                  Point(1,0), Point(1,1)};

    int id = dir ? grad_id: grad_id+4;
    int max_val = edgeMap.at<uchar>(new_pt.x+offsets[id].x, new_pt.y+offsets[id].y);
    int max_id = id;
    if (max_val < edgeMap.at<uchar>(new_pt.x+offsets[(id+1) % 8].x,
                                    new_pt.y+offsets[(id+1) % 8].y)) {
        max_id = (id+1) % 8;
    } else if (max_val < edgeMap.at<uchar>(new_pt.x+offsets[(id+7) % 8].x,
                                    new_pt.y+offsets[(id+7) % 8].y)) {
        max_id = (id+7) % 8;
    }
    new_pt += offsets[max_id];

    return static_cast<int>(edgeMap.at<uchar>(new_pt.x, new_pt.y));
}

inline bool isValidSeed( const int x, const int y, const Mat& edgeMap, const Mat gradMap[],
                 const int gradThreshold=10) {
  if (edgeMap.at<uchar>(x,y) < gradThreshold) return false;
  GRAD_ID grad_id = getGradID(gradMap, Point(x,y));
  switch (grad_id) {
    case HORIZONTAL:  if (edgeMap.at<uchar>(x,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x,y-1) < gradThreshold) {
                        return false;
                      } break;
    case UPHILL:      if (edgeMap.at<uchar>(x-1,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x+1,y-1) < gradThreshold) {
                        return false;
                      } break;
    case VERTICAL:    if (edgeMap.at<uchar>(x+1,y) < gradThreshold or
                          edgeMap.at<uchar>(x-1,y) < gradThreshold) {
                        return false;
                      } break;
    case DOWNHILL:    if (edgeMap.at<uchar>(x+1,y+1) < gradThreshold or
                          edgeMap.at<uchar>(x-1,y-1) < gradThreshold) {
                        return false;
                      } break;
  }

  return true;
}

inline void shiftSeed(Point& seed, const Mat& edgeMap, const Mat gradMap[]) {
  GRAD_ID grad_id  = getGradID(gradMap, seed);
  uchar edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
  switch (grad_id) {
    case HORIZONTAL: while (edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y)) {
                      seed.x++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y)) {
                      seed.x--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case UPHILL:    while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y-1)) {
                      seed.x--; seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while(edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y+1)) {
                      seed.x++; seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case VERTICAL:  while (edgeVal < edgeMap.at<uchar>(seed.x, seed.y+1)) {
                      seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x, seed.y-1)) {
                      seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
    case DOWNHILL:  while (edgeVal < edgeMap.at<uchar>(seed.x-1, seed.y+1)) {
                      seed.x--; seed.y++;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    while (edgeVal < edgeMap.at<uchar>(seed.x+1, seed.y-1)) {
                      seed.x++; seed.y--;
                      edgeVal = edgeMap.at<uchar>(seed.x, seed.y);
                    }
                    break;
  }
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

bool exploreContour(const Point& seed, Mat& edgeMap, Mat gradMap[],
                    std::vector<cv::Point>& contour, const int explore_length=8) {
    // add early failure detection
    GRAD_ID grad_id;
    cv::Point new_pt = Point(seed);
    int edgeVal;// = edgeMsp.at<uchar>(seed.x, seed.y);
    for (int i = 0; i < explore_length; i++) {
      grad_id = getGradID(gradMap, new_pt);
      edgeVal = moveAlongContour(new_pt, grad_id, edgeMap, true);
      if (edgeVal < 15) return false;
      contour.push_back(new_pt);
    }
    std::reverse(std::begin(contour), std::end(contour));
    contour.push_back(seed);
    new_pt = Point(seed);
    for (int i = 0; i < explore_length; i++) {
      grad_id = getGradID(gradMap, new_pt);
      edgeVal = moveAlongContour(new_pt, grad_id, edgeMap, false);
      if (edgeVal < 15) return false;
      contour.push_back(new_pt);
    }
    return true;
}

void linearFit(const vec_iter_t& start, const vec_iter_t& end, cv::Point3f& params) {
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
}

void expandSeed(const Point& seed, Mat& edgeMap, Mat gradMap[]) {
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
  std::printf("--- Running Contour Detection ---\n" );
  clock_t t = clock();
  suppressNoise(img_gray, img_gray);
  t = clock()-t;
  std::printf("I applied gaussian filter in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  t = clock();
  Mat edgeMap;
  Mat gradMap[] = {Mat(img_gray.rows, img_gray.cols, CV_16S),
                   Mat(img_gray.rows, img_gray.cols, CV_16S)};
  computeEdgeAndGradMap(img_gray, edgeMap, gradMap);
  t = clock()-t;
  std::printf("I computed edge and gradient maps in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  std::vector<cv::Point> seeds;
  t = clock();
  extractSeeds(edgeMap, gradMap, seeds);
  t = clock() - t;
  std::printf("I extracted %d seeds in %f ms\n", static_cast<int>(seeds.size()),
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  // visualization for debugging
  Mat color;
  cv::cvtColor(edgeMap, color, cv::COLOR_GRAY2BGR);

  std::vector<cv::Point> contour;
  // seeds.clear();
  // seeds.push_back(cv::Point(349,65));
  t = clock();
  for (auto& seed: seeds) {
    shiftSeed(seed, edgeMap, gradMap);
    contour.clear();
    int contourLength = 12;
    if (!exploreContour(seed, edgeMap,gradMap, contour, contourLength)) {
      continue;
    }
  //   cv::Point3f lineParams;
  //   linearFit(contour.begin(), contour.end(), lineParams);
  //   for (auto& pt: contour) {
  //     color.at<Vec3b>(pt.x,pt.y)[0] = 0;
  //     color.at<Vec3b>(pt.x,pt.y)[1] = 0;
  //     color.at<Vec3b>(pt.x,pt.y)[2] = 255;
  //   }
  //   int new_x = static_cast<int>(12*lineParams.x);
  //   int new_y = static_cast<int>(12*lineParams.y);
  //   cv::Point start(contour[contourLength].y, contour[contourLength].x);
  // cv::line(color, start, start+cv::Point(new_y, new_x), Scalar(0,255,0));
  // break;
  }
  t = clock() - t;
  std::printf("I explored contours in %f ms\n",
              ((float)t)/CLOCKS_PER_SEC*1000.0);

  namedWindow("Seed Map", WINDOW_NORMAL );
  resizeWindow("Seed Map", 1000, 800);
  imshow("Seed Map", color );
  waitKey(0);
}


// add early failure detection to expandContour

#endif
