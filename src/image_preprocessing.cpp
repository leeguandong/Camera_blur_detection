#include "image_preprocessing.h"
#define BLOCK 60

double blur_detect(Mat img) {
	Mat gray, dst, abs_dst;
	Scalar mean, stddev;

	//GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//Laplacian(gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	Laplacian(gray, dst, CV_64F);
	convertScaleAbs(dst, abs_dst);

	meanStdDev(abs_dst, mean, stddev, Mat());
	double variance = stddev.val[0] * stddev.val[0];

	return variance;
}

void blur_detect_sobel(Mat img, float& cast) {
	Mat gray, dst;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	Sobel(gray, dst, CV_16U, 1, 1);

	cast = mean(dst)[0];
}


double blur_detect_fft(Mat img) {
	Mat gray;

	cvtColor(img, gray, COLOR_BGR2GRAY);

	int cx = gray.cols / 2;
	int cy = gray.rows / 2;

	// go float
	Mat fImage;
	gray.convertTo(fImage, CV_32F);

	// fft
	Mat fourierTransform;
	dft(fImage, fourierTransform, DFT_SCALE | DFT_COMPLEX_OUTPUT);

	// center low frequencies in the middle by shuffling the quadrants  
	Mat q0(fourierTransform, Rect(0, 0, cx, cy)); // Top-Left create a ROI per quadrant
	Mat q1(fourierTransform, Rect(cx, 0, cx, cy)); // Top-Right
	Mat q2(fourierTransform, Rect(0, cy, cx, cy)); // Botttom-Left
	Mat q3(fourierTransform, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp; // swap quadrants(Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp); // swap quadrants(Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Block the low frequencies
	// #define BLOCK could also be a argument on the command line of course
	fourierTransform(Rect(cx - BLOCK, cy - BLOCK, 2 * BLOCK, 2 * BLOCK)).setTo(0);

	// shuffle the quadrants to their original position
	Mat orgFFT;
	fourierTransform.copyTo(orgFFT);
	Mat p0(orgFFT, Rect(0, 0, cx, cy));  // Top-Left create a ROI per quadrants
	Mat p1(orgFFT, Rect(cx, 0, cx, cy)); // Top-Right
	Mat p2(orgFFT, Rect(0, cy, cx, cy)); // Bottom-Left
	Mat p3(orgFFT, Rect(cx, cy, cx, cy)); // Bottom-Right

	p0.copyTo(tmp);
	p3.copyTo(p0);
	tmp.copyTo(p3);

	p1.copyTo(tmp); // swap quadrant(Top-Right with Bottom-Left)
	p2.copyTo(p1);
	tmp.copyTo(p2);

	Mat invFFT;
	Mat logFFT;
	double minVal, maxVal;

	dft(orgFFT, invFFT, DFT_INVERSE | DFT_REAL_OUTPUT);

	invFFT = abs(invFFT);
	minMaxLoc(invFFT, &minVal, &maxVal, NULL, NULL);

	// check for impossible values
	if (maxVal <= 0.0) {
		return double(-1);
	}

	log(invFFT, logFFT);
	logFFT *= 20;
	Scalar result = mean(logFFT);

	Mat finalImage;
	logFFT.convertTo(finalImage, CV_8U); // back to 8-bits
	imwrite("E:/common_tools/blur/blur_steganalysis/data/fft_img.jpg", finalImage);

	namedWindow("fft_img", WINDOW_AUTOSIZE);
	imshow("fft_img", finalImage);
	waitKey(0);
	destroyAllWindows();

	return double(result.val[0]);
}


void getHaarWavelet(const Mat& src, Mat& dst) {
	int height = src.size().height;
	int width = src.size().width;
	dst.create(height, width, CV_32F);

	Mat horizontal = Mat::zeros(height, width, CV_32F);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width / 2; j++) {
			float meanPixel = (src.at<float>(i, 2 * j) + src.at<float>(i, 2 * j + 1)) / 2;
			horizontal.at<float>(i, j) = meanPixel;
			horizontal.at<float>(i, j + width / 2) = src.at<float>(i, 2 * j) - meanPixel;
		}
	}

	for (int i = 0; i < height / 2; i++) {
		for (int j = 0; j < width; j++) {
			float meanPixel = (horizontal.at<float>(2 * i, j) + horizontal.at<float>(2 * i + 1, j)) / 2;
			dst.at<float>(i, j) = meanPixel;
			dst.at<float>(i + height / 2, j) = horizontal.at<float>(2 * i, j) - meanPixel;
		}
	}

	horizontal.release();
}


void getEmax(const Mat& src, Mat& dst, int scale) {
	int height = src.size().height;
	int width = src.size().width;
	int h_scaled = height / scale;
	int w_scaled = width / scale;
	dst.create(h_scaled, w_scaled, CV_32F);

	for (int i = 0; i < h_scaled; i++) {
		for (int j = 0; j < w_scaled; j++) {
			double maxValue;
			minMaxLoc(src(Rect(scale * j, scale * i, scale, scale)), NULL, &maxValue);
			dst.at<float>(i, j) = float(maxValue);
		}
	}
}


double blur_detect_haarwavelet(Mat img) {
	float threshold = 35;
	float MinZero = 0.05;

	Mat gray;

	cvtColor(img, gray, COLOR_BGR2GRAY);

	int height0 = gray.size().height;
	int width0 = gray.size().width;
	gray.convertTo(gray, CV_32F);

	int height = ceilf(float(height0) / 16) * 16;
	int width = ceilf(float(width0) / 16) * 16;
	Mat img_ = Mat::zeros(height, width, CV_32F);
	Mat temp = img_(Rect(0, 0, width0, height0));
	gray.copyTo(img_(Rect(0, 0, width0, height0)));

	// 1.Algorithmm 1:HWT for edge detection
	// step1 (Harr wavelet transform)
	Mat level1;
	getHaarWavelet(img_, level1);
	Mat level2;
	getHaarWavelet(level1(Rect(0, 0, width / 2, height / 2)), level2);
	Mat level3;
	getHaarWavelet(level2(Rect(0, 0, width / 4, height / 4)), level3);

	// step2
	Mat HL1, LH1, HH1, Emap1;
	pow(level1(Rect(width / 2, 0, width / 2, height / 2)), 2.0, HL1);
	pow(level1(Rect(0, height / 2, width / 2, height / 2)), 2.0, LH1);
	pow(level1(Rect(width / 2, height / 2, width / 2, height / 2)), 2.0, HH1);
	sqrt(HL1 + LH1 + HH1, Emap1);

	Mat HL2, LH2, HH2, Emap2;
	pow(level2(Rect(width / 4, 0, width / 4, height / 4)), 2.0, HL2);
	pow(level2(Rect(0, height / 4, width / 4, height / 4)), 2.0, LH2);
	pow(level2(Rect(width / 4, height / 4, width / 4, height / 4)), 2.0, HH2);
	sqrt(HL2 + LH2 + HH2, Emap2);

	Mat HL3, LH3, HH3, Emap3;
	pow(level3(Rect(width / 8, 0, width / 8, height / 8)), 2.0, HL3);
	pow(level3(Rect(0, height / 8, width / 8, height / 8)), 2.0, LH3);
	pow(level3(Rect(width / 8, height / 8, width / 8, height / 8)), 2.0, HH3);
	sqrt(HL3 + LH3 + HH3, Emap3);

	// step3
	Mat Emax1, Emax2, Emax3;
	getEmax(Emap1, Emax1, 8);
	getEmax(Emap2, Emax2, 4);
	getEmax(Emap3, Emax3, 2);

	// Algorithm 2: blur detection scheme
	// Step1(Algorithm 1)
	// step2
	int m = Emax1.size().height;
	int n = Emax2.size().width;
	int Nedge = 0;
	Mat Eedge = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (Emax1.at<float>(i, j) > threshold || Emax2.at<float>(i, j) > threshold
				|| Emax3.at<float>(i, j) > threshold) {
				++Nedge;
				Eedge.at<float>(i, j) = 1.0;
			}
		}
	}

	// step3 (Rule2)
	int Nda = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float tempEmax2 = Emax2.at<float>(i, j);
			if (Eedge.at<float>(i, j) > 0.1 && Emax1.at<float>(i, j) > tempEmax2
				&& tempEmax2 > Emax3.at<float>(i, j)) {
				++Nda;
			}
		}
	}

	// step4(Rule3,4)
	int Nrg = 0;
	Mat Eedge_Gstep_Roof = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float tempEmax1 = Emax1.at<float>(i, j);
			float tempEmax2 = Emax2.at<float>(i, j);
			float tempEmax3 = Emax3.at<float>(i, j);
			if (Eedge.at<float>(i, j) > 0.1 &&
				(tempEmax1<tempEmax2 && tempEmax2<tempEmax3
					|| tempEmax2>tempEmax1 && tempEmax2>tempEmax3)) {
				++Nrg;
				Eedge_Gstep_Roof.at<float>(i, j) = 1.0;
			}
		}
	}

	// step5(Rule5)
	int Nbrg = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (Eedge_Gstep_Roof.at<float>(i, j) > 0.1 && Emax1.at<float>(i, j) < threshold) {
				++Nbrg;
			}
		}
	}

	// step6
	float Per = float(Nda) / Nedge;
	int unblured = 0;
	if (Per > MinZero) {
		unblured = 1;
	}

	// step7
	float BlurExtent = float(Nbrg) / Nrg;

	cout << "Num of edge points: " << Nedge << endl;
	cout << "Num of Dirac and Astep: " << Nda << endl;
	cout << "Num of Roof and Gstep: " << Nrg << endl;
	cout << "Num of Roof and Gstep lost sharp: " << Nbrg << endl;
	cout << "BlurExtent: " << BlurExtent << endl;

	return unblured;
}


double blackscreen_detect(Mat img) {
	Mat gray;
	int dark_sum = 0;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	for (int i = 0; i < gray.rows; i++) {
		for (int j = 0; j < gray.cols;) {
			if (float(gray.at<uchar>(i, j)) < 20)
				dark_sum++;
		}
	}
	double mean = dark_sum / float(gray.total());
	return mean;
}

double brightness_detect(Mat img) {
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	float sum = 0, Ma = 0;
	int hist[256] = { 0 };

	for (int i = 0; i < gray.rows; i++) {
		for (int j = 0; j < gray.cols; j++) {
			sum += float(gray.at<uchar>(i, j) - 128);//在计算过程中，考虑128为亮点均值点
			hist[gray.at<uchar>(i, j)]++;
		}
	}

	float da = sum / float(gray.total()); // 

	for (int i = 0; i < 256; i++) {
		Ma += abs(i - 128 - da) * hist[i]; // 计算偏离中心值
	}

	Ma /= float(gray.total());
	float cast = abs(da) / abs(Ma);
	// cast 计算出的偏差值，小于1.0表示比较正常，大于1.0表示存在亮度异常；
	// 当cast异常时，da大于0表示过亮，da小于0表示过暗
	if (cast > 1)
		return da;
	else
		return 0;
}

void color_detect(Mat img, float& cast, float& da, float& db) {
	Mat lab;
	cvtColor(img, lab, COLOR_BGR2Lab);

	float a = 0, b = 0;
	int HistA[256] = { 0 }, HistB[256] = { 0 };

	for (int i = 0; i < lab.rows; i++) {
		for (int j = 0; j < lab.cols; j++) {
			a += lab.at<Vec3b>(i, j)[1];// A
			b += lab.at<Vec3b>(i, j)[2];

			HistA[lab.at<Vec3b>(i, j)[1]]++;
			HistB[lab.at<Vec3b>(i, j)[2]]++;
		}
	}

	da = a / float(lab.total()) - 128;
	db = b / float(lab.total()) - 128;

	float Ma = 0, Mb = 0;
	for (int i = 0; i < 256; i++) {
		Ma += abs(i - 128 - da) * HistA[i];
		Mb += abs(i - 128 - db) * HistB[i];
	}

	Ma /= float(lab.total());
	Mb /= float(lab.total());

	cast = sqrt(da * da + db * db) / sqrt(Ma * Ma + Mb * Mb);
	// cast 计算出的偏差值，小于1.5表示比较正常，大于1.5表示存在色偏。
	// da   红/绿色偏估计值，da大于0，表示偏红；da小于0表示偏绿。
	// db   黄/蓝色偏估计值，db大于0，表示偏黄；db小于0表示偏蓝。
}


void separation_foreground(Mat img, Mat& mask) {
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(500, 16, true);

	pMOG2->apply(img, mask);
}




