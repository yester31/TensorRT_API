#pragma once

// calibration 데이터 전처리 함수
// https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface
cv::Mat preprocess_img_cali(cv::Mat& img, int input_w, int input_h);

// 특정 폴더내 파일 이름 리스트 출력 함수
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

// cpp 전처리 함수 
//void preprocessImg(cv::Mat& img, int newh, int neww);