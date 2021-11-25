#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <string>
#include <io.h>				// access
#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static const int INPUT_C = 3;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		Weights wt{ DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;

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

// Creat the engine using only the API and not any parser.
void createEngine( unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
	std::cout << "==== model build start ====" << std::endl << std::endl;
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::map<std::string, Weights> weightMap = loadWeights("../VGG11_py/vgg11.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{  INPUT_H, INPUT_W, INPUT_C });
	assert(data);

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	preprocess_layer->setName("preprocess_layer"); // layer 이름 설정

	IConvolutionLayer* conv1 = network->addConvolutionNd(*preprocess_layer->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["features.0.weight"], weightMap["features.0.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStrideNd(DimsHW{ 2, 2 });

	conv1 = network->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{ 3, 3 }, weightMap["features.3.weight"], weightMap["features.3.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStrideNd(DimsHW{ 2, 2 });

	conv1 = network->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{ 3, 3 }, weightMap["features.6.weight"], weightMap["features.6.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{ 3, 3 }, weightMap["features.8.weight"], weightMap["features.8.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStrideNd(DimsHW{ 2, 2 });

	conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.11.weight"], weightMap["features.11.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.13.weight"], weightMap["features.13.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStrideNd(DimsHW{ 2, 2 });

	conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.16.weight"], weightMap["features.16.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.18.weight"], weightMap["features.18.bias"]);
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStrideNd(DimsHW{ 2, 2 });

	IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
	relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
	fc1 = network->addFullyConnected(*relu1->getOutput(0), 4096, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);
	relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
	fc1 = network->addFullyConnected(*relu1->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);

	fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*fc1->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 23);

	std::cout << "Building engine, please wait for a while..." << std::endl;
	IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
	std::cout << "==== model build done ====" << std::endl << std::endl;

	std::cout << "==== model selialize start ====" << std::endl << std::endl;
	std::ofstream p(engineFileName, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl << std::endl;
	}
	p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
	std::cout << "==== model selialize done ====" << std::endl << std::endl;

	engine->destroy();
	network->destroy();
	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
}

int main()
{
	// 변수 선언 
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	bool serialize = false;			// Serialize 강제화 시키기(true 엔진 파일 생성)
	char engineFileName[] = "vgg11";

	char engine_file_path[256];
	sprintf(engine_file_path, "../Engine/%s.engine", engineFileName);

	// 1) engine file 만들기 
	// 강제 만들기 true면 무조건 다시 만들기
	// 강제 만들기 false면, engine 파일 있으면 안만들고 
	//					   engine 파일 없으면 만듬
	bool exist_engine = false;
	if ((access(engine_file_path, 0) != -1)) {
		exist_engine = true;
	}

	if (!((serialize == false)/*Serialize 강제화 값*/ == (exist_engine == true) /*vgg.engine 파일이 있는지 유무*/)) {
		std::cout << "===== Create Engine file =====" << std::endl << std::endl; // 새로운 엔진 생성
		IBuilder* builder = createInferBuilder(gLogger);
		IBuilderConfig* config = builder->createBuilderConfig();
		createEngine(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path); // *** Trt 모델 만들기 ***
		builder->destroy();
		config->destroy();
		std::cout << "===== Create Engine file =====" << std::endl << std::endl; // 새로운 엔진 생성 완료
	}

	// 2) engine file 로드 하기 
	char *trtModelStream{ nullptr };// 저장된 스트림을 저장할 변수
	size_t size{ 0 };
	std::cout << "===== Engine file load =====" << std::endl << std::endl;
	std::ifstream file(engine_file_path, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		file.read(trtModelStream, size);
		file.close();
	}
	else {
		std::cout << "[ERROR] Engine file load error" << std::endl;
	}

	// 3) file에서 로드한 stream으로 tensorrt model 엔진 생성
	std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	IExecutionContext* context = engine->createExecutionContext();
	delete[] trtModelStream;

	void* buffers[2];
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// GPU에서 입력과 출력으로 사용할 메모리 공간할당
	CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
	CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

	// 4) 입력으로 사용할 이미지 준비하기
	std::string img_dir = "../TestDate/";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
		std::cerr << "[ERROR] Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
	}
	cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> outputs(OUTPUT_SIZE);
	for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
		cv::Mat ori_img = cv::imread(file_names[idx]);
		cv::resize(ori_img, img, img.size()); // input size로 리사이즈
		memcpy(input.data(), img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
	}
	std::cout << "===== input load done =====" << std::endl << std::endl;

	uint64_t dur_time = 0;
	uint64_t iter_count = 100;

	// CUDA 스트림 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// 5) Inference 수행  
	for (int i = 0; i < iter_count; i++) {
		// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
		context->enqueue(maxBatchSize, buffers, stream, nullptr);
		CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
		dur_time += dur;
		//std::cout << dur << " milliseconds" << std::endl;
	}

	// 6) 결과 출력
	std::cout << "==================================================" << std::endl;
	std::cout << "===============" << engineFileName << "===============" << std::endl;
	std::cout << iter_count  << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
	int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	std::cout << "Index : "<< max_index << ", Probability : " << outputs[max_index] << ", Class Name : " << class_names[max_index] <<  std::endl;
	std::cout << "==================================================" << std::endl;

	// Release stream and buffers ...
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}