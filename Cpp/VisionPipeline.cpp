#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Transformations.hpp"
#include "EdgeDrawing.hpp"
#include "LineDrawing.hpp"
#include "ContourDetection.hpp"

using namespace cv;
//g++ $(pkg-config --cflags --libs opencv) -w VisionPipeline.cpp -o VisionPipeline

int main(int argc, char** argv )
{
  int num_files = 23;
  for (int i = 0; i != num_files; i++) {
    Mat image = imread("../Pics/blocks/" + std::to_string(i) +".JPG" , 1);
    Mat image_gray;
    cvtColor( image, image_gray, CV_BGR2GRAY );

    // Mat BW = Mat(image.rows, image.cols, CV_8UC1);
    // Mat RG = Mat(image.rows, image.cols, CV_8UC1);
    // Mat YB = Mat(image.rows, image.cols, CV_8UC1);
    // convertToBWRGYB(image, BW, RG, YB);
    Mat contourMap = Mat::zeros(image_gray.rows, image_gray.cols, CV_8UC3);
    extractContours(image_gray, contourMap);

    hconcat(image, contourMap, image);

    // imwrite( "../Pics/results/contour" + std::to_string(i) +".JPG" , color );

    namedWindow("Seed Map", WINDOW_NORMAL );
    moveWindow("Seed Map", 0,30);
    resizeWindow("Seed Map", 800,600);
    imshow("Seed Map", image );
    waitKey(0);
  }
  return 0;
}
