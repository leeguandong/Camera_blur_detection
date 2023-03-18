#include "image_preprocessing.h"
#include "utils.h"

int getdir(string dir, vector<string>& files) {
	DIR* dp;
	struct dirent* dirp;
	if ((dp = opendir(dir.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening" << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) {
		files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

void printText(Mat src, string text) {
	int fontFace = FONT_HERSHEY_PLAIN;
	putText(src, text, Point(20, src.rows / 10), fontFace, 1.2f, Scalar(200, 0, 0), 2);
}

void show_result(Mat src, string text_, size_t index, double value, double threshold) {
	string text;
	stringstream ss;
	stringstream save_path; //字符串流

	namedWindow("result", WINDOW_AUTOSIZE);

	ss << "image is ";
	if (value > threshold) // 850  Guassian效果不好
		ss << text_;
	else
		ss << "not " << text_;
	ss << " value: " << value << endl;

	getline(ss, text); // 将ss赋给text
	printText(src, text);

	imshow("result", src);
	waitKey(0);
	destroyAllWindows();

	save_path << "E:/common_tools/blur/blur_steganalysis/data/" << text_ << "_" << index << ".jpg";
	string save_path_ = save_path.str();
	imwrite(save_path_, src);
}

Mat VisClassification(Mat& img, fastdeploy::vision::ClassifyResult& result,
	int top_k, float score_threshold, float font_size) {
	int h = img.rows;
	int w = img.cols;
	auto vis_img = img.clone();
	int h_sep = h / 30;
	int w_sep = w / 10;
	if (top_k > result.scores.size()) {
		top_k = result.scores.size();
	}
	for (int i = 0; i < top_k; ++i) {
		if (result.scores[i] < score_threshold) {
			continue;
		}
		std::string id = std::to_string(result.label_ids[i]);
		std::string score = std::to_string(result.scores[i]);
		if (score.size() > 4) {
			score = score.substr(0, 4);
		}
		std::string text = id + "," + score;
		int font = FONT_HERSHEY_COMPLEX;
		Point origin;
		origin.x = w_sep;
		origin.y = h_sep * (i + 1);
		putText(vis_img, text, origin, font, font_size, Scalar(255, 255, 255), 1);
	}
	return vis_img;
}



