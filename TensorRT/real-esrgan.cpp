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
#include <io.h>				//access
#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 448;
static const int INPUT_C = 3;
static const int OUT_SCALE = 4;
//static const int OUTPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W * 64;

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

void show_dims(ITensor* tensor) 
{
	std::cout << "=== show dims ===" << std::endl;
	int dims = tensor->getDimensions().nbDims;
	std::cout << "size :: " << dims << std::endl;
	for (int i = 0; i < dims; i++) {
		std::cout << tensor->getDimensions().d[i] << std::endl;
	}
}

ITensor* residualDenseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname) 
{

	IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv_1->setStrideNd(DimsHW{ 1, 1 });
	conv_1->setPaddingNd(DimsHW{ 1, 1 });
	IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kLEAKY_RELU);
	leaky_relu_1->setAlpha(0.2);
	ITensor* x1 = leaky_relu_1->getOutput(0);

	ITensor* concat_input2[] = { x, x1 };
	IConcatenationLayer* concat2 = network->addConcatenation(concat_input2, 2);
	concat2->setAxis(0);
	IConvolutionLayer* conv_2 = network->addConvolutionNd(*concat2->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv_2->setStrideNd(DimsHW{ 1, 1 });
	conv_2->setPaddingNd(DimsHW{ 1, 1 });
	IActivationLayer* leaky_relu_2 = network->addActivation(*conv_2->getOutput(0), ActivationType::kLEAKY_RELU);
	leaky_relu_2->setAlpha(0.2);
	ITensor* x2 = leaky_relu_2->getOutput(0);

	ITensor* concat_input3[] = { x, x1, x2 };
	IConcatenationLayer* concat3 = network->addConcatenation(concat_input3, 3);
	concat3->setAxis(0);
	IConvolutionLayer* conv_3 = network->addConvolutionNd(*concat3->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv3.weight"], weightMap[lname + ".conv3.bias"]);
	conv_3->setStrideNd(DimsHW{ 1, 1 });
	conv_3->setPaddingNd(DimsHW{ 1, 1 });
	IActivationLayer* leaky_relu_3 = network->addActivation(*conv_3->getOutput(0), ActivationType::kLEAKY_RELU);
	leaky_relu_3->setAlpha(0.2);
	ITensor* x3 = leaky_relu_3->getOutput(0);

	ITensor* concat_input4[] = { x, x1, x2, x3 };
	IConcatenationLayer* concat4 = network->addConcatenation(concat_input4, 4);
	concat4->setAxis(0);
	IConvolutionLayer* conv_4 = network->addConvolutionNd(*concat4->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv4.weight"], weightMap[lname + ".conv4.bias"]);
	conv_4->setStrideNd(DimsHW{ 1, 1 });
	conv_4->setPaddingNd(DimsHW{ 1, 1 });
	IActivationLayer* leaky_relu_4 = network->addActivation(*conv_4->getOutput(0), ActivationType::kLEAKY_RELU);
	leaky_relu_4->setAlpha(0.2);
	ITensor* x4 = leaky_relu_4->getOutput(0);

	ITensor* concat_input5[] = { x, x1, x2, x3, x4 };
	IConcatenationLayer* concat5 = network->addConcatenation(concat_input5, 5);
	concat5->setAxis(0);
	IConvolutionLayer* conv_5 = network->addConvolutionNd(*concat5->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap[lname + ".conv5.weight"], weightMap[lname + ".conv5.bias"]);
	conv_5->setStrideNd(DimsHW{ 1, 1 });
	conv_5->setPaddingNd(DimsHW{ 1, 1 });
	ITensor* x5 = conv_5->getOutput(0);

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*scval = 0.2;
	Weights scale{ DataType::kFLOAT, scval, 1 };
	float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*shval = 0.0;
	Weights shift{ DataType::kFLOAT, shval, 1 };
	float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*pval = 1.0;
	Weights power{ DataType::kFLOAT, pval, 1 };

	IScaleLayer* scaled = network->addScale(*x5, ScaleMode::kUNIFORM, shift, scale, power);
	IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
	return ew1->getOutput(0);
}


ITensor* RRDB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname)
{
	ITensor* out = residualDenseBlock(network, weightMap, x, lname + ".rdb1");
	out = residualDenseBlock(network, weightMap, out, lname + ".rdb2");
	out = residualDenseBlock(network, weightMap, out, lname + ".rdb3");

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*scval = 0.2;
	Weights scale{ DataType::kFLOAT, scval, 1 };
	float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*shval = 0.0;
	Weights shift{ DataType::kFLOAT, shval, 1 };
	float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
	*pval = 1.0;
	Weights power{ DataType::kFLOAT, pval, 1 };

	IScaleLayer* scaled = network->addScale(*out, ScaleMode::kUNIFORM, shift, scale, power);
	IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
	return ew1->getOutput(0);
}

// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
	std::cout << "==== model build start ====" << std::endl << std::endl;
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::map<std::string, Weights> weightMap = loadWeights("../Real-ESRGAN_py/real-esrgan.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
	assert(data);

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	preprocess_layer->setName("preprocess_layer"); // layer 이름 설정
	ITensor* prep = preprocess_layer->getOutput(0);

	// conv_first
	IConvolutionLayer* conv_first = network->addConvolutionNd(*prep, 64, DimsHW{ 3, 3 }, weightMap["conv_first.weight"], weightMap["conv_first.bias"]);
	conv_first->setStrideNd(DimsHW{ 1, 1 });
	conv_first->setPaddingNd(DimsHW{ 1, 1 });
	conv_first->setName("conv_first"); // layer 이름 설정
	ITensor* feat = conv_first->getOutput(0);

	// conv_body
	for (int idx = 0; idx < 23; idx++) 
	{
		feat = RRDB(network, weightMap, feat, "body." + std::to_string(idx));
	}

	IConvolutionLayer* conv_body = network->addConvolutionNd(*feat, 64, DimsHW{ 3, 3 }, weightMap["conv_body.weight"], weightMap["conv_body.bias"]);
	conv_body->setStrideNd(DimsHW{ 1, 1 });
	conv_body->setPaddingNd(DimsHW{ 1, 1 });
	conv_body->setName("conv_body"); // layer 이름 설정
	feat = conv_body->getOutput(0);


	//upsample

	//IResizeLayer* interpolate_nearest = network->addResize(*feat); // 1,1024,60,80 -> 1,1024,120,160
	//float sclaes[] = { 1, 1, 2, 2 };
	//interpolate_nearest->setScales(sclaes, 4);
	//interpolate_nearest->setResizeMode(ResizeMode::kNEAREST);
	//pspupsample_bilinear->setCoordinateTransformation(ResizeCoordinateTransformation::kALIGN_CORNERS);

	ITensor* final_tensor = feat;
	show_dims(final_tensor);
	final_tensor->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*final_tensor);

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 20);

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
	bool serialize = true;			// Serialize 강제화 시키기(true 엔진 파일 생성)
	char engineFileName[] = "real-esrgan";

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

	if (!((serialize == false)/*Serialize 강제화 값*/ && (exist_engine == true) /*resnet18.engine 파일이 있는지 유무*/)) {
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
	std::string img_dir = "../TestData3/";
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
	uint64_t iter_count = 1;

	// CUDA 스트림 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// warm-up
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
	context->enqueue(maxBatchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

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

	tofile(outputs, "../Validation_py/c"); // 결과값 파일로 출력
	//../Validation_py/valide_preproc.py 에서 결과 비교 ㄱㄱ

	// 6) 결과 출력
	std::cout << "==================================================" << std::endl;
	std::cout << "===============" << engineFileName << "===============" << std::endl;
	std::cout << iter_count << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
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