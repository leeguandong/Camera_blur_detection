#ifndef IMAGE_PREPROCESSING
#define IMAGE_PREPROCESSING

#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

double blur_detect(Mat img);

void blur_detect_sobel(Mat img, float& cast);

double blur_detect_fft(Mat img);

double blur_detect_haarwavelet(Mat img);

/*
将图像灰度化，检测偏暗的像素占总体比例，推断黑屏,速度更快
*/
double blackscreen_detect(Mat img);

/*
计算图片在灰度图上均值和方差,当存在亮度异常时，均值会偏离均值点，可以假设为128，方差也会偏小，
通过计算灰度图的均值和方差，就可以评估是否存在过曝光和曝光不足。
*/
double brightness_detect(Mat img);

/*
将RGB图像转到到LAB空间，其中L代表图像亮度，A表示图像红/绿分量，B表示图像黄/蓝分量。通常存在色偏的图像，
在A和B分量上的均值会偏离原点很远，方差也会偏小；通过计算图像在A和B分量上的均值和方差，就可评估图像是否存在色偏。
*/
void color_detect(Mat img, float& cast, float& da, float& db);

/*
如果检测的图像背景正常就是虚的，那可以在检测前提取图像前景，对劲对象前景进行模糊检测
混合高斯模糊为基础的前景/背景分割算法
*/
void separation_foreground(Mat img, Mat& mask);

#endif // !IMAGE_PREPROCESSING
