#ifndef CONTOURDETECTION_INCLUDE
#define CONTOURDETECTION_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include "Transformations.hpp"

using namespace cv;

void extractSeeds(Mat& img_gray, std::vector<Point>& dst, int size=10, int threshold = 20) {
  Mat region_frame;
  Point current_pt;
  Point offset = Point(size,size);
  double minVal;
  double maxVal;
  Point minLoc;
  Point maxLoc;

  for(int y=0; y<=(img_gray.rows - size); y+=size)
  {
      for(int x=0; x<=(img_gray.cols - size); x+=size)
      {
          current_pt.x = x;
          current_pt.y = y;
          Rect region = Rect(current_pt, current_pt+offset);
          region_frame = img_gray(region);
          minMaxLoc( region_frame, &minVal, &maxVal, &minLoc, &maxLoc );
          if (maxVal >= threshold) {
            dst.push_back(Point(maxLoc.y+y, maxLoc.x+x));
          }
      }
  }
}

void displaySeedLocations(Mat& img_gray) {
  Mat edges;
  edgeDetector(img_gray, edges);
  std::vector<Point> seeds;
  extractSeeds(edges, seeds, 10);
  std::printf("There are %d seeds\n", static_cast<int>(seeds.size()));
  Mat seedMap = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
  for (const auto& seed: seeds) {
    seedMap.at<uchar>(seed.x,seed.y) = 255;
  }

  namedWindow("Seed Map", WINDOW_NORMAL );
  resizeWindow("Seed Map", 1000, 800);
  imshow("Seed Map", edges );
  waitKey(0);
}


#endif
