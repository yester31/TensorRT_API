#include <iostream>
#include <iterator>
#include <fstream>
#include <opencv2/dnn/dnn.hpp>
#include "calibrator.h"
#include "cuda_runtime_api.h"
#include "common.hpp"		
#include <opencv2/opencv.hpp>

// CUDA RUNTIME API 에러 체크를 위한 매크로 함수 정의
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, int process_type, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
	: batchsize_(batchsize)
	, input_w_(input_w)
	, input_h_(input_h)
	, img_idx_(0)
	, process_type_(process_type)
	, img_dir_(img_dir)
	, calib_table_name_(calib_table_name)
	, input_blob_name_(input_blob_name)
	, read_cache_(read_cache)
{
	input_count_ = 3 * input_w * input_h * batchsize;
	input_size_ = 3 * input_w * input_h;
	CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(uint8_t)));
	read_files_in_dir(img_dir, img_files_);

}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
	CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
	return batchsize_;
} 

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
	if (img_idx_ + batchsize_ > (int)img_files_.size()) {
		return false;
	}

	if (process_type_ == 0) { // vgg, resnet, detr (with preprocess layer) 
		std::vector<uint8_t> input_imgs_(input_count_, 0);
		cv::Mat img(input_h_, input_w_, CV_8UC3);
		for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
			std::cout << img_files_[i] << "  " << i << std::endl;
			cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
			if (temp.empty()) {
				std::cerr << "Fatal error: image cannot open!" << std::endl;
				return false;
			}
			cv::resize(temp, img, img.size(), cv::INTER_LINEAR);
			memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img.data, input_size_);
		}
		img_idx_ += batchsize_;
		CHECK(cudaMemcpy(device_input_, input_imgs_.data(), input_count_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
	}else if (process_type_ == 1) {  // unet (with preprocess layer) 
		std::vector<uint8_t> input_imgs_(input_count_, 0);
		cv::Mat img(input_h_, input_w_, CV_8UC3);
		for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
			std::cout << img_files_[i] << "  " << i << std::endl;
			cv::Mat ori_img = cv::imread(img_dir_ + img_files_[i]);
			if (ori_img.empty()) {
				std::cerr << "Fatal error: image cannot open!" << std::endl;
				return false;
			}
			int ori_w = ori_img.cols;
			int ori_h = ori_img.rows;
			if (ori_h == ori_w) { // 입력이미지가 정사각형일 경우
				cv::Mat img_r(input_h_, input_w_, CV_8UC3);
				cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 모델 사이즈로 리사이즈
				memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img_r.data, input_size_);
			}
			else {
				int new_h, new_w;
				if (ori_w >= ori_h) {
					new_h = (int)(ori_h * ((float)input_w_ / ori_w));
					new_w = input_w_;
				}
				else {
					new_h = input_h_;
					new_w = (int)(ori_w * ((float)input_h_ / ori_h));
				}
				cv::Mat img_r(new_h, new_w, CV_8UC3);
				cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR);
				int tb = (int)((input_h_ - new_h) / 2);
				int bb = ((new_h % 2) == 1) ? tb + 1 : tb;
				int lb = (int)((input_w_ - new_w) / 2);
				int rb = ((new_w % 2) == 1) ? lb + 1 : lb;
				cv::Mat img_p(input_h_, input_w_, CV_8UC3);
				cv::copyMakeBorder(img_r, img_p, tb, bb, lb, rb, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
				memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img_p.data, input_size_);
			}
		}
		img_idx_ += batchsize_;
		CHECK(cudaMemcpy(device_input_, input_imgs_.data(), input_count_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
	}
	else if (process_type_ == 2) {
		std::vector<uint8_t> input_imgs_(input_count_, 0);
		cv::Mat img(input_h_, input_w_, CV_8UC3);
		for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
			std::cout << img_files_[i] << "  " << i << std::endl;
			cv::Mat ori_img = cv::imread(img_dir_ + img_files_[i]);
			if (ori_img.empty()) {
				std::cerr << "Fatal error: image cannot open!" << std::endl;
				return false;
			}
			int ori_w = ori_img.cols;
			int ori_h = ori_img.rows;

			if (ori_h == ori_w) { // 입력이미지가 정사각형일 경우
				cv::Mat img_r(input_h_, input_w_, CV_8UC3);
				cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 모델 사이즈로 리사이즈
				memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img_r.data, input_size_);
			}
			else {
				float ratio = std::min(((float)input_h_ / ori_w), ((float)input_w_ / ori_h));
				int	new_h = (int)(round(ori_h * ratio));
				int	new_w = (int)(round(ori_w * ratio));
				cv::Mat img_r(new_h, new_w, CV_8UC3);
				cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 정합성 일치
				//int dh = (INPUT_H - new_h) % 32;
				//int dw = (INPUT_W - new_w) % 32;
				int dh = (input_h_ - new_h);
				int dw = (input_w_ - new_w);
				int tb = (int)round(((float)dh / 2) - 0.1);
				int bb = (int)round(((float)dh / 2) + 0.1);
				int lb = (int)round(((float)dw / 2) - 0.1);
				int rb = (int)round(((float)dw / 2) + 0.1);
				cv::Mat img_p((new_h + dh), (new_w + dw), CV_8UC3);
				cv::copyMakeBorder(img_r, img_p, tb, bb, lb, rb, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
				memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img_p.data, input_size_);
			}
		}
		img_idx_ += batchsize_;
		CHECK(cudaMemcpy(device_input_, input_imgs_.data(), input_count_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
	}
	else { // 
		std::cerr << "Fatal error: pre-preprocess type is wrong!" << std::endl;
		return false;
	}

	assert(!strcmp(names[0], input_blob_name_));
	bindings[0] = device_input_;
	return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
	std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
	calib_cache_.clear();
	std::ifstream input(calib_table_name_, std::ios::binary);
	input >> std::noskipws;
	if (read_cache_ && input.good())
	{
		std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
	}
	length = calib_cache_.size();
	return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
	std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
	std::ofstream output(calib_table_name_, std::ios::binary);
	output.write(reinterpret_cast<const char*>(cache), length);
}

