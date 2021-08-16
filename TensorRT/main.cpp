#include "utils.h"
#include "NvInfer.h"
#include "logging.h"
#include "cuda_runtime_api.h"

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

using namespace nvinfer1;

static Logger gLogger;

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


void main() {

	// 0. 이미지경로 로드
	std::string img_dir = "../data";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) {
		std::cerr << "data search error" << std::endl;
	}

	// 1. 이미지 데이터 로드
	int batch_size = 1;
	int input_width = 640;
	int input_height = 640;
	Mat img(input_height, input_width, CV_8UC3);
	Mat ori_img;
	vector<uint8_t> input(batch_size * input_height * input_width * 3);

	for (int idx = 0; idx < file_names.size(); idx++) {
		Mat ori_img = imread(file_names[idx]);
		resize(ori_img, img, img.size());
		memcpy(input.data(), img.data, batch_size * input_height * input_width * 3);
	}

	std::vector<float> output(OUTPUT_SIZE);

	std::cout << "Create Engine file" << std::endl; // 새로운 엔진 생성

	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();
	ICudaEngine* engine{ nullptr };
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 

	//==========================================================================================
	INetworkDefinition* network = builder->createNetworkV2(0U);
	// Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
	nvinfer1::DataType dt = nvinfer1::DataType::kFLOAT;
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W }); // [N,C,H,W]
	assert(data);

	//IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
	//fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	//std::cout << "set name out" << std::endl;
	//network->markOutput(*fc1->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 22);
	engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "build out" << std::endl;

	// Don't need the network any more
	network->destroy();

	IHostMemory* modelStream{ nullptr };	// 저장하기 위해 serialzie된 모델 스트림 
	modelStream = engine->serialize();
	std::ofstream p("newmodel.engine", std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

	//==========================================================================================
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
	CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync((uint8_t*)buffers[inputIndex], input.data(), maxBatchSize * 3 * INPUT_H * INPUT_W , cudaMemcpyHostToDevice, stream));

	context->enqueue(maxBatchSize, buffers, stream, nullptr);

	CHECK(cudaMemcpyAsync(output.data(), (uint8_t*)buffers[outputIndex], maxBatchSize * OUTPUT_SIZE , cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// print result
	int max_index = max_element(output.begin(), output.end()) - output.begin();
	std::cout << max_index << " , " << class_names[max_index] << " , " << output[max_index] << std::endl;


	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	builder->destroy();
	engine->destroy();
	context->destroy();
	config->destroy();
	modelStream->destroy();
}