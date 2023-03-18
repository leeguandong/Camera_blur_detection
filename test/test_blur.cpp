#include "image_preprocessing.h"
#include "fastdeploy/vision.h"
#include "picodet_postprocess.h"
#include "utils.h"
#include <stdio.h>

using namespace std;
using namespace cv;

int main() {

	String pattern = "E:/common_tools/blur/steganalysis_preprocessing/data/success";
	std::string det_model_file =
		"E:/common_tools/blur/blur_steganalysis/data/picodet_l_320_coco_lcnet/model.pdmodel";
	std::string det_params_file =
		"E:/common_tools/blur/blur_steganalysis/data/picodet_l_320_coco_lcnet/model.pdiparams";
	std::string det_config_file =
		"E:/common_tools/blur/blur_steganalysis/data/picodet_l_320_coco_lcnet/infer_cfg.yml";

	std::string cls_model_file =
		"E:/common_tools/blur/blur_steganalysis/data/PPLCNetV2_base_infer/inference.pdmodel";
	std::string cls_params_file =
		"E:/common_tools/blur/blur_steganalysis/data/PPLCNetV2_base_infer/inference.pdiparams";
	std::string cls_config_file =
		"E:/common_tools/blur/blur_steganalysis/data/PPLCNetV2_base_infer/infer_cfg.yml";

	auto option = fastdeploy::RuntimeOption();
	//option.UseCpu();
	//option.UseOrtBackend();
	option.UseOpenVINOBackend();
	//option.UsePaddleBackend();

	//option.UseGpu();
	//option.UseOrtBackend();
	//option.UseTrtBackend();
	//option.UsePaddleBackend();

	//�ƶ��� arm
	//option.UseLiteBackend(); //Paddlelite

	// ��Ե��
	//option.UseTimVX();
	//option.UseRKNPU2();
	//option.UseKunlunXin();
	//option.UseAscend();

	// Picodet ��ppdet�е�ģ��
	auto det_model = fastdeploy::vision::detection::PicoDet(
		det_model_file, det_params_file, det_config_file, option);

	auto cls_model = fastdeploy::vision::classification::PPLCNetv2(
		cls_model_file, cls_params_file, cls_config_file, option);

	if (!det_model.Initialized()) {
		std::cerr << "Failed to initialize." << std::endl;
		return -1;
	}

	if (!cls_model.Initialized()) {
		std::cerr << "Failed to initialize." << std::endl;
		return -1;
	}

	vector<cv::String> fn;
	glob(pattern, fn, false);
	size_t count = fn.size();
	for (size_t i = 0; i < count; i++) {
		Mat img = imread(fn[i]);

		// ���ģ��
		fastdeploy::vision::DetectionResult result; // 
		if (!det_model.Predict(&img, &result)) {
			std::cerr << "Failed to predict." << std::endl;
			return -1;
		}
		cout << "nms֮ǰ��" << result.Str() << endl;
		
		NMS(&result, float(0.3));

		// visual 
		float score_threshold = 0.5;
		int line_size = 2;
		float font_size = 0.5;
		auto vis_img = fastdeploy::vision::Visualize::VisDetection(img.clone(),
			result, score_threshold, line_size, font_size);

		stringstream save_path;
		save_path << "E:/common_tools/blur/blur_steganalysis/data/picodet_" << i << ".jpg";
		string save_path_ = save_path.str();
		imwrite(save_path_, vis_img);

		namedWindow("vis_img", WINDOW_AUTOSIZE);
		imshow("vis_img", vis_img);
		waitKey(0);
		destroyAllWindows();

		cout << "nms֮��" << result.Str() << endl;
		// ԭͼ�����ü����ü�֮���ͼ���д�ͳ�㷨�ж�
		//cout << result->boxes.size() << endl;
		float xmin = (&result)->boxes[0][0]; // ȡ���Ŷȷ�����ߵ�һ����ǰ��������
		float ymin = (&result)->boxes[0][1];
		float xmax = (&result)->boxes[0][2];
		float ymax = (&result)->boxes[0][3];
		float overlap_w = std::max(0.0f, xmax - xmin);
		float overlap_h = std::max(0.0f, ymax - ymin);
		Rect rect(xmin, ymin, overlap_w, overlap_h); //�������Ͻ����꣬����������
		Mat img_roi = img(rect);
		imshow("img_roi", img_roi);
		waitKey(0);

		// ǰ�����ָ�
		//Mat mask;
		//separation_foreground(img, mask);
		//imshow("img", mask);
		//waitKey(0);

		// �������
		//double blackscreen = blackscreen_detect(img);
		//show_result(img.clone(), "blackscreen", size_t(i), double(blackscreen), double(0.85));

		// ɫƫ���  ����
		float cast = 0, da = 0, db = 0;
		color_detect(img_roi, cast, da, db);
		if (cast > 1.5) {
			//Mat img1 = img.clone();
			show_result(img_roi.clone(), "red(green)", size_t(i), double(da), double(0));
			show_result(img_roi.clone(), "yellow(blue)", size_t(i), double(db), double(0));
		}

		// ���ȼ��
		double brightness = brightness_detect(img_roi);
		if (brightness != 0) //�����쳣����ֵ��Ϊ0������0��ʾ������С��0��ʾ����
			show_result(img_roi.clone(), "too bright", size_t(i), double(brightness), double(0));

		// ������˹����
		double laplacian = blur_detect(img_roi);
		show_result(img_roi.clone(), "blur", size_t(i), double(laplacian), double(100));

		// sobel����
		float sobel = 0;
		blur_detect_sobel(img_roi, sobel);
		show_result(img_roi.clone(), "clear_sobel", size_t(i), double(sobel), double(1.5)); // С��1.5����ģ��

		// fft
		double fft = blur_detect_fft(img_roi);
		show_result(img_roi.clone(), "blur_fft", size_t(i), double(fft), double(1));

		// haarwavelet
		double haarwavelet = blur_detect_haarwavelet(img_roi);
		string text;
		stringstream ss;
		stringstream save_path_haarwavelet; //�ַ�����

		namedWindow("result", WINDOW_AUTOSIZE);

		ss << "image is ";
		if (haarwavelet == 1)
			ss << "clear";
		else
			ss << "blured ";

		getline(ss, text); // ��ss����text
		printText(img_roi, text);

		imshow("result", img_roi);
		waitKey(0);
		destroyAllWindows();

		save_path_haarwavelet << "E:/common_tools/blur/blur_steganalysis/data/haarwavelet" << "_" << size_t(i) << ".jpg";
		string save_path_haarwavelet_ = save_path_haarwavelet.str();
		imwrite(save_path_haarwavelet_, img_roi);

		// ģ������ģ���ж�
		fastdeploy::vision::ClassifyResult cls_result;
		if (!cls_model.Predict(img_roi, &cls_result)) {
			std::cerr << "Failed to predict." << std::endl;
			return -1;
		}

		// print res
		std::cout << cls_result.Str() << std::endl;
		//int label = (&cls_result)->label_ids[0];
		//float scores = (&cls_result)->scores[0];
		float cls_score_threshold = 0.9;
		int top_k = 1;
		std::vector<std::string> labels = { "clear","fuzzy" };
		//auto vis_cls_img = VisClassification(img_roi.clone(),
		//	cls_result, top_k, cls_score_threshold, font_size);
		auto vis_cls_img = fastdeploy::vision::VisClassification(
			img_roi.clone(), cls_result, labels, top_k, cls_score_threshold, font_size
		);

		stringstream save_path_cls;
		save_path_cls << "E:/common_tools/blur/blur_steganalysis/data/pplcnetv2_" << i << ".jpg";
		string save_path_cls_ = save_path_cls.str();
		imwrite(save_path_cls_, vis_cls_img);

		namedWindow("vis_cls_img", WINDOW_AUTOSIZE);
		imshow("vis_cls_img", vis_cls_img);
		waitKey(0);
		destroyAllWindows();
	}

	//string dir = string("E:/common_tools/blur/steganalysis_preprocessing/data/success");
	//vector<string> files = vector<string>();
	//getdir(dir,files);

	//for (unsigned int i = 0; i < files.size(); i++) {
	//	cout << files[i] << endl;
	//	img_path = dir + files[i];
	//}
	return 0;
}










