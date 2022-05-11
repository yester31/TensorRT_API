#include "utils.hpp"

// 파일 이름 가져오기(DFS) window용
// full path name
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive)
{
	_finddata_t file_info;
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
		if (recursive) {
			if (file_info.attrib & _A_SUBDIR)//check whtether it is a sub direcotry or a file
			{
				if (file_name != "." && file_name != "..")
				{
					std::string sub_folder_path = folder_path + "//" + file_name;
					SearchFile(sub_folder_path, file_names);
					std::cout << "a sub_folder path: " << sub_folder_path << std::endl;
				}
			}
			else
			{
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
		else {
			if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
			{
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}

void valueCheck(std::vector<float>& Input, int IN, int IC, int IH, int IW, bool one) {
	std::cout << "===== valueCheck func =====" << std::endl;
	if (one) IN = 1;
	int tot = IN * IC * IH * IW;
	if (Input.size() != tot) {
		if (tot == 1)
		{
			IN = Input.size();
		}
		else {
			return;
		}
	}
	int N_offset = IC * IH * IW;
	int C_offset, H_offset, W_offset, g_idx;
	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		C_offset = ⁠n_idx * N_offset;
		for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
			H_offset = ⁠c_idx * IW * IH + C_offset;
			for (int ⁠h_idx = 0; ⁠h_idx < IH; ⁠h_idx++) {
				W_offset = ⁠h_idx * IW + H_offset;
				for (int w_idx = 0; w_idx < IW; w_idx++) {
					g_idx = w_idx + W_offset;
					std::cout << std::setw(5) << Input[g_idx] << " ";
				}std::cout << std::endl;
			}std::cout << std::endl; std::cout << std::endl;
		}
	}
}

void initTensor(std::vector<float>& output, float start, float step)
{
	std::cout << "===== InitTensor func (scalar or step)=====" << std::endl;
	float count = start;
	for (int i = 0; i < output.size(); i++) {
		output[i] = count;
		count += step;
	}
}

void initTensor(std::vector<float>& output, std::string random, float min, float max)
{
	std::cout << "===== InitTensor func (random value) =====" << std::endl;

	for (int i = 0; i < output.size(); i++) {
		output[i] = min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
	}
}

void initTensor(std::vector<float>& output, int N, int C, int H, int W, float start, float step) {
	std::cout << "===== scalarTensor func =====" << std::endl;
	std::cout << "Tensor[" << N << "][" << C << "][" << H << "][" << W << "]" << std::endl << std::endl;
	int tot_size = N * C * H * W;
	if (output.size() != tot_size)
		output.resize(tot_size);
	initTensor(output, start, step);
}



int argMax(std::vector<float> &output) {

	return max_element(output.begin(), output.end()) - output.begin();
}
//std::cout << "index : "<< argMax(output) << " , label name : " << class_names[argMax(output) ] << " , prob : " << output[argMax(output) ] << std::endl;



std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
{
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, nvinfer1::Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = nvinfer1::DataType::kFLOAT;

		// Load blob
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
	}

	return weightMap;
}

void show_dims(nvinfer1::ITensor* tensor)
{
	std::cout << "=== show dims ===" << std::endl;
	int dims = tensor->getDimensions().nbDims;
	std::cout << "size :: " << dims << std::endl;
	for (int i = 0; i < dims; i++) {
		std::cout << tensor->getDimensions().d[i] << std::endl;
	}
}
