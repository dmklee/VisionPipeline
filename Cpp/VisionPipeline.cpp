#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctime>
#include <algorithm>
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
  gp << "set terminal wxt size 900,650" << std::endl;

  int num_files = 30;
  int img_id = 5;

  Mat image = imread("../../Pics/blocks/" + std::to_string(img_id) +".JPG" , 1);
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

  bool isEdge = true;
  bool isColor = true;
  bool newImage = false;
  int max_length = 50;
  int DRIFT_TOL = 500;
  int DECAY_RATE = 90;
  Mat toShow;
  std::vector<cv::Point> edgels;
  std::vector<cv::Point> corners;
  std::vector< std::pair<double, double> > orientations;
  std::vector< std::pair<double, double> > d_orientations;
  std::vector< std::pair<double, double> > mu_data;
  std::vector< std::pair<double, double> > std_hi_data;
  std::vector< std::pair<double, double> > std_lo_data;
  std::vector< std::pair<double, double> > frontier_data;

  namedWindow(img_name, WINDOW_NORMAL );
  setMouseCallback(img_name, CallBackFunc, &click_data);
  moveWindow(img_name, 0,30);
  resizeWindow(img_name, 700,900);
  createTrackbar( "contour length", img_name, &max_length, 500, 0 );
  createTrackbar( "drift tol (*1000)", img_name, &DRIFT_TOL, 1000, 0 );
  createTrackbar( "decay rate (*100)", img_name, &DECAY_RATE, 100, 0 );
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
        corners.clear();
        orientations.clear();
        d_orientations.clear();
        mu_data.clear();
        std_hi_data.clear();
        std_lo_data.clear();
        frontier_data.clear();
        Point tmp = Point(click_data.pt.y, click_data.pt.x);
        extractCorners(image_gray, tmp, edgels, orientations);
        if (edgels.size() == 0) {
          break;
        }
        if (edgels.size() > max_length) edgels.resize(max_length);
        if (orientations.size() > max_length) orientations.resize(max_length);


        //////////////////////////////////////
        double mu = 0.0;
        double m2 = 0.1;
        double delta, delta_2;
        const int f_size = 5;
        int corner_status = 0;

        double front_mu = 0.0;
        for (int i = 0; i != orientations.size(); i++) {
          delta = get<1>(orientations[i]) - mu;
          mu += (delta)/ (i + 1);
          delta_2 =  get<1>(orientations[i]) - mu;
          m2 += delta*delta_2;
          mu_data.push_back(std::make_pair(i, mu));
          std_hi_data.push_back(std::make_pair(i, mu + sqrt(m2/(i+1))));
          std_lo_data.push_back(std::make_pair(i, mu - sqrt(m2/(i+1))));

          if (i < orientations.size() - f_size) {
            for (int j=0; j != f_size; j ++) {
              front_mu += std::get<1>(orientations[i+j]);
            }
            front_mu /= f_size;
            frontier_data.push_back(std::make_pair(i, front_mu ));

            if (abs(front_mu - mu) > sqrt(m2/(i+1)) ) {
              if (front_mu < mu) {
                std::cout << "too low\n";
              } else {
                std::cout << "too high\n";
              }
              corners.push_back(edgels[i+f_size]);
              break;
            }
          }
        }

        gp << "set multiplot layout 1,1" << std::endl;
        gp << "plot" << gp.file1d(orientations) << "with lp pt 7 lt rgb \"blue\" title 'angle', ";
        gp << gp.file1d(mu_data) << "with lp pt 7 lt rgb \"black\" dashtype 18 notitle, ";
        gp << gp.file1d(std_hi_data) << "with lines lt rgb \"black\" dashtype 16 notitle, ";
        gp << gp.file1d(std_lo_data) << "with lines lt rgb \"black\" dashtype 16 notitle, ";
        gp << gp.file1d(frontier_data) << "with lines lt rgb \"red\" dashtype 16 notitle";
        gp << std::endl;
        // gp << "plot" << gp.file1d(d_orientations) << "with lp pt 7 lt rgb \"red\" title 'd angle', ";
        // gp << gp.file1d(mu_data) << "with lp pt 7 lt rgb \"black\" dashtype 18 title 'avg'";
        // gp << ", " << gp.file1d(std_data) << "with lines lt rgb \"black\" dashtype 16 notitle";
        // gp << ", " << gp.file1d(error) << "with lines lt rgb \"violet\" dashtype 20 notitle";
        // gp << std::endl;
        gp << "unset multiplot" << std::endl;

        // std::cout << "\n(" << std::get<1>(orientations[0]);
        // for (int i = 1; i != orientations.size(); i++) {
        //   std::cout << "," << std::get<1>(orientations[i]);
        // }
        // std::cout << ")\n";
        break;
    }


    if (newImage) {
      image = imread("../../Pics/blocks/" + std::to_string(img_id) +".JPG" , 1);
      // GaussianBlur(image,image,Size(5,5),sigma/10.0);
      cv::cvtColor( image, image_gray, cv::COLOR_BGR2GRAY );
      cv::cvtColor(image_gray, c_gray, cv::COLOR_GRAY2BGR);
      edgeDetector(image_gray, edgeMap);
      cv::cvtColor(edgeMap, c_edgeMap, cv::COLOR_GRAY2BGR);
      newImage = false;
      click_data.valid = false;
      edgels.clear();
      corners.clear();
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
    for (const auto& e: corners) {
      toShow.at<Vec3b>(e.x, e.y)[0] = 0;
      toShow.at<Vec3b>(e.x, e.y)[1] = 0;
      toShow.at<Vec3b>(e.x, e.y)[2] = 255;
    }

    imshow(img_name, toShow );

  } // end for loop
  return 0;
}
