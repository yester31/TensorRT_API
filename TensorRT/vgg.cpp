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
#include <io.h>		//access
#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	


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

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

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
void createEngine(ICudaEngine* engine, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
	INetworkDefinition* network = builder->createNetworkV2(0U);

	// Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);

	std::map<std::string, Weights> weightMap = loadWeights("../vgg/vgg.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 3, 3 }, weightMap["features.0.weight"], weightMap["features.0.bias"]);
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
	std::cout << "set name out" << std::endl;
	network->markOutput(*fc1->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 22);
	engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "build out" << std::endl;

	// Don't need the network any more
	network->destroy();

	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
}

void doInference(ICudaEngine* engine, float* input, float* output, int batchSize)
{
	IExecutionContext* context = engine->createExecutionContext();
	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context->enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	context->destroy();

}

int vgg()
{
	// 변수 선언 
	char *trtModelStream{ nullptr };		// 저장된 스트림을 저장할 변수
	IHostMemory* modelStream{ nullptr };	// 저장하기 위해 serialzie된 모델 스트림 
	ICudaEngine* engine{ nullptr };
	size_t size{ 0 };
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	bool serialize = false;			// Serialize 강제화 시키기(true 엔진 파일 생성)
	char strPath[] = { "vgg.engine" };
	
	if (access(strPath, 0) == 0 /*vgg.engine 파일이 있는지 유무*/  && !serialize /*Serialize 강제화 값*/) {
		
		std::cout << "Engine file exists" << std::endl; // 엔진파일이 존재 하므로 로드 작업 수행
		std::ifstream file("vgg.engine", std::ios::binary);	
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}

		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);

		engine = runtime->deserializeCudaEngine(trtModelStream, size); // 파일에서 로드한 스트림을 이용하여 엔진생성
		assert(engine != nullptr);
		runtime->destroy();
		delete[] trtModelStream;
	}
	else {
		std::cout << "Create Engine file" << std::endl; // 새로운 엔진 생성

		IBuilder* builder = createInferBuilder(gLogger);
		IBuilderConfig* config = builder->createBuilderConfig();

		createEngine(engine, maxBatchSize, builder, config, DataType::kFLOAT); // *** Trt 모델 만들기 ***
		assert(engine != nullptr);

		modelStream = engine->serialize();
		assert(modelStream != nullptr);

		std::ofstream p("vgg.engine", std::ios::binary);
		if (!p) {
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

		builder->destroy();
		config->destroy();
		modelStream->destroy();
	}
	

	std::vector<float> input(3 * INPUT_H * INPUT_W);
	std::ifstream ifs("../data/input", std::ios::binary); //bgr -> rgb, nhwc -> nchw
	if (ifs.is_open())
		ifs.read((char*)input.data(), input.size() * sizeof(float));
	ifs.close();
	std::vector<float> input_f(input.begin(), input.end());

	std::vector<float> outputs(OUTPUT_SIZE);


	// Run inference
	for (int i = 0; i < 1; i++) {
		auto start = std::chrono::system_clock::now();		
		doInference(engine, input_f.data(), outputs.data(), maxBatchSize);		
		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	}

	// print result
	int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	std::cout << max_index << " , " << class_names[max_index] << " , " << outputs[max_index] << std::endl;

	// Destroy the engine
	engine->destroy();

	return 0;
}