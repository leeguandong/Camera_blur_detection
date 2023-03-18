#pragma once
#ifndef  UTILS
#define UTILS
#include <iostream>
#include <string>

#include "fastdeploy/vision.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int getdir(string, vector<string>&);

void printText(Mat src, string text);

void show_result(Mat src, string text, size_t index, double value, double threshold);

Mat VisClassification(Mat& img, fastdeploy::vision::ClassifyResult& result,
	int top_k = 5, float score_threshold = 0.0f, float font_size = 0.5f);


#endif // ! UTILS


