#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <io.h>	// access
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "NvInferRuntime.h"
#include "NvInferPlugin.h"


// 파일 이름 가져오기(DFS) window용
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false);

// vector 값 출력
void valueCheck(std::vector<float>& Input, int IN = 1, int IC = 1, int IH = 1, int IW = 1, bool one = false);

// 데이터 초기화 함수 (스칼라, 등차수열)
void initTensor(std::vector<float>& output, float start = 1, float step = 0);

// 데이터 초기화 함수 (랜덤 생성)
void initTensor(std::vector<float>& output, std::string random, float min = -10.f, float max = 10.f);

// 데이터 초기화 함수 (차원값 입력)
void initTensor(std::vector<float>& output, int N, int C, int H, int W, float start = 1, float step = 0);

// 데이터 바이너리 파일로 생성 (serialize) 
template<class T>
void tofile(std::vector<T> &Buffer, std::string fname = "../Validation_py/C_Tensor") {
	std::ofstream fs(fname, std::ios::binary);
	if (fs.is_open())
		fs.write((const char*)Buffer.data(), Buffer.size() * sizeof(T));
	fs.close();
	std::cout << "Done! file production to " << fname << std::endl;
}
// 데이터 바이너리 파일 로드 (unserialize) 
// 사용 예) 
// fromfile(input, "../Unet_py/input_data"); // python 전처리 결과 로드
template<class T>
void fromfile(std::vector<T>& Buffer, std::string fname = "../Validation_py/C_Tensor") {
	std::ifstream ifs(fname, std::ios::binary);
	if (ifs.is_open())
		ifs.read((char*)Buffer.data(), Buffer.size() * sizeof(T));
	ifs.close();
	std::cout << "Done! file load from " << fname << std::endl;
}
// 최대값 인덱스 출력 함수
// 사용 예) 
// std::cout << "index : "<< argMax(output) << " , label name : " << class_names[argMax(output) ] << " , prob : " << output[argMax(output) ] << std::endl;
int argMax(std::vector<float> &output);

// colors table
//std::vector<std::vector<int>> colors_table{ {244,67,54},{233,30,99},{156,39,176},{103,58,183},{63,81,181},{33,150,243},{3,169,244},
//{0,188,212}, {0,150,136}, {76,175,80}, {139,195,74}, {205,220,57}, {255,235,59}, {255,193,7},
//{255,152,0}, {255,87,34}, {121,85,72}, {158,158,158}, {96,125,139} };

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

// Print Tensor dimensions information
void show_dims(nvinfer1::ITensor* tensor);