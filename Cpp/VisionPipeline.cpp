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
  int num_files = 1;
  int start = 4;
  for (int i=start; i < start+num_files; i++ ) {

    // Mat image = imread("../Pics/blocks/" + std::to_string(i) +".JPG" , 1);
    Mat image = imread("../Pics/occlusion4.png" , 1);
    Mat image_gray;
    cvtColor( image, image_gray, CV_BGR2GRAY );

    // Mat BW = Mat(image.rows, image.cols, CV_8UC1);
    // Mat RG = Mat(image.rows, image.cols, CV_8UC1);
    // Mat YB = Mat(image.rows, image.cols, CV_8UC1);
    // convertToBWRGYB(image, BW, RG, YB);
    Mat contourMap;
    // extractContours(image_gray, contourMap);
    extractCorners(image_gray, contourMap);
    // runLineDrawing(image_gray, contourMap);

    // hconcat(image, contourMap, image);

    // imwrite( "../Pics/results/contour" + std::to_string(i) +".JPG" , color );

    std::string img_name = "img_" + std::to_string(i);
    namedWindow(img_name, WINDOW_NORMAL );
    moveWindow(img_name, 0,30);
    resizeWindow(img_name, 800,1000);
    imshow(img_name, contourMap );
    waitKey(0);
  }
  return 0;
}
