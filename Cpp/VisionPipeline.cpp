#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;

int main(int argc, char** argv )
{
  Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  Mat image = imread("../Pics/circle.png", 1 );
  Mat image_gray;
  cvtColor( image, image_gray, CV_BGR2GRAY );

  std::clock_t start;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  start = std::clock();
  Sobel( image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

  convertScaleAbs( grad_x, abs_grad_x );

  Sobel( image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  // namedWindow("Display Image", WINDOW_AUTOSIZE );
  // imshow("Display Image", grad);
  // waitKey(0);
  return 0;
}
