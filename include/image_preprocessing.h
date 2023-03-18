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
��ͼ��ҶȻ������ƫ��������ռ����������ƶϺ���,�ٶȸ���
*/
double blackscreen_detect(Mat img);

/*
����ͼƬ�ڻҶ�ͼ�Ͼ�ֵ�ͷ���,�����������쳣ʱ����ֵ��ƫ���ֵ�㣬���Լ���Ϊ128������Ҳ��ƫС��
ͨ������Ҷ�ͼ�ľ�ֵ�ͷ���Ϳ��������Ƿ���ڹ��ع���عⲻ�㡣
*/
double brightness_detect(Mat img);

/*
��RGBͼ��ת����LAB�ռ䣬����L����ͼ�����ȣ�A��ʾͼ���/�̷�����B��ʾͼ���/��������ͨ������ɫƫ��ͼ��
��A��B�����ϵľ�ֵ��ƫ��ԭ���Զ������Ҳ��ƫС��ͨ������ͼ����A��B�����ϵľ�ֵ�ͷ���Ϳ�����ͼ���Ƿ����ɫƫ��
*/
void color_detect(Mat img, float& cast, float& da, float& db);

/*
�������ͼ�񱳾�����������ģ��ǿ����ڼ��ǰ��ȡͼ��ǰ�����Ծ�����ǰ������ģ�����
��ϸ�˹ģ��Ϊ������ǰ��/�����ָ��㷨
*/
void separation_foreground(Mat img, Mat& mask);

#endif // !IMAGE_PREPROCESSING
