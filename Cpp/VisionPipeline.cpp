#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"
#include "ContourDetection.hpp"

#include <cmath>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream/gnuplot-iostream.h"

using namespace cv;
using namespace std;
//g++ $(pkg-config --cflags --libs opencv) -w VisionPipeline.cpp -o VisionPipelin

struct ImageMetaData {
  Mat bw_img;
  Mat bgr_img;
  std::string window_name;
};

struct MetaData {
  Point pt = Point(0,0);
  bool valid = false;
};

void CallBackFunc(int event, int x, int y, int flags, void* param)
{
  MetaData* data = (MetaData*) param;
  if  ( event == EVENT_LBUTTONDOWN )
  {
    data->pt.x = x;
    data->pt.y = y;
    data->valid = true;
  }
}

int main(int argc, char** argv )
{
  Gnuplot gp;
  std::vector<std::pair<double, double> > z;
  z.push_back(std::make_pair(0.,1.));
  z.push_back(std::make_pair(1.,1.));


  int num_files = 23;
  int start = 0;
  Mat image = imread("../../Pics/blocks/" + std::to_string(0) +".JPG" , 1);
  // Mat image = imread("../Pics/occlusion4.png" , 1);
  Mat image_gray;
  cv::cvtColor( image, image_gray, cv::COLOR_BGR2GRAY);
  Mat edgeMap;
  edgeDetector(image_gray, edgeMap);

  Mat c_gray;
  Mat c_edgeMap;
  cv::cvtColor(image_gray, c_gray, cv::COLOR_GRAY2BGR);
  cv::cvtColor(edgeMap, c_edgeMap, cv::COLOR_GRAY2BGR);

  std::string img_name = "img";//+ std::to_string(i);
  MetaData click_data;

  bool isEdge = false;
  bool isColor = true;
  bool newImage = false;
  int img_id = 2;
  int loDiff = 10;
  Mat toShow;
  std::vector<cv::Point> edgels;
  std::vector< std::pair<double, double> > orientations;

  namedWindow(img_name, WINDOW_NORMAL );
  setMouseCallback(img_name, CallBackFunc, &click_data);
  moveWindow(img_name, 0,30);
  resizeWindow(img_name, 700,900);
  createTrackbar( "lo_diff", img_name, &loDiff, 255, 0 );
  imshow(img_name, image );
  for (;;) {
    char c = (char)waitKey(0);
    // std::cout << (int)c << "\n";
    if (c == 27) {
      // escape key
      break;
    }
    switch (c)
    {
      case 'e':
        isEdge = !isEdge;
        break;
      case 'c':
        isColor = !isColor;
        break;
      case 46:
        //right arrow
        img_id = (1+img_id) % num_files;
        newImage = true;
        break;
      case 44:
        //left arrow
        img_id = (img_id+num_files-1) % num_files;
        newImage = true;
        break;
      case 13:
        // enter key
        // here we want to run contour detection
        edgels.clear();
        orientations.clear();
        Point tmp = Point(click_data.pt.y, click_data.pt.x);
        extractCorners(image_gray, tmp, edgels, orientations);
        gp << "plot" << gp.file1d(orientations) << "with lines" << std::endl;
    }
    if (newImage) {
      image = imread("../../Pics/blocks/" + std::to_string(img_id) +".JPG" , 1);
      cv::cvtColor( image, image_gray, cv::COLOR_BGR2GRAY );
      cv::cvtColor(image_gray, c_gray, cv::COLOR_GRAY2BGR);
      edgeDetector(image_gray, edgeMap);
      cv::cvtColor(edgeMap, c_edgeMap, cv::COLOR_GRAY2BGR);
      newImage = false;
      click_data.valid = false;
      edgels.clear();
    }

    if (isEdge) {
      c_edgeMap.copyTo(toShow);
    } else if (isColor) {
      image.copyTo(toShow);
    } else {
      c_gray.copyTo(toShow);
    }
    if (click_data.valid) {
      cv::circle(toShow, click_data.pt, 1, Scalar(0,0,255));
    }
    for (const auto& e: edgels) {
      toShow.at<Vec3b>(e.x, e.y)[0] = 255;
      toShow.at<Vec3b>(e.x, e.y)[1] = 0;
      toShow.at<Vec3b>(e.x, e.y)[2] = 0;
    }

    imshow(img_name, toShow );

  } // end for loop
  return 0;
}
