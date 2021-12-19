#include <io.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// calibration 데이터 전처리 함수
// https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface
cv::Mat preprocess_img_cali(cv::Mat& img, int input_w, int input_h) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	return out;
}

// 특정 폴더내 파일 이름 리스트 출력 함수
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
	_finddata_t file_info;
	const std::string folder_path = p_dir_name;
	std::string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);

	if (handle == -1)
	{
		std::cerr << "folder path not exist: " << folder_path << std::endl;
		return -1;
	}
	do
	{
		std::string file_name = file_info.name;
		if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
		{
			//std::string file_path = folder_path + "/" + file_name;
			file_names.push_back(file_name);
		}

	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}

//void preprocessImg(cv::Mat& img, int newh, int neww) {
//	// convert to rgb
//	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//	cv::resize(img, img, cv::Size(neww, newh));
//	img.convertTo(img, CV_32FC3);
//	img /= 255;
//	img -= cv::Scalar(0.485, 0.456, 0.406);
//	img /= cv::Scalar(0.229, 0.224, 0.225);
//}

