#ifndef VISUALIZATION_INCLUDE
#define VISUALIZATION_INCLUDE

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

void showImage(Mat& img_gray);

void showImageWithDots(Mat& img_gray, std::vector<std::actor<int,2> >& pointList);

#endif
