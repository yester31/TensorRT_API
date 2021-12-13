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
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !
static const int CHECK_COUNT = 3;

struct alignas(float) Detection {
	//center_x center_y w h
	float bbox[4];
	float conf;  // bbox_conf * cls_conf
	float class_id;
};

struct YoloKernel
{
	int width;
	int height;
	float anchors[CHECK_COUNT * 2];
};

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int CLASS_NUM = 80;
static const int MAX_OUTPUT_BBOX_COUNT = 1000;
static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const int INPUT_C = 3;
static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

// yolov5s 
static const int  gd = 0.33;
static const int  gw = 0.50;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";



static int get_width(int x, float gw, int divisor = 8) {
	return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
	if (x == 1) return 1;
	int r = round(x * gd);
	if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
		--r;
	}
	return std::max<int>(r, 1);
}

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file){
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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);
ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname);
ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname);
ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname);
ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname);
std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname);
IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets);


// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
	std::cout << "==== model build start ====" << std::endl << std::endl;
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::map<std::string, Weights> weightMap = loadWeights("../yolov5s_py/yolov5s.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
	assert(data);

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	//preprocess_layer->setName("preprocess_layer"); // layer 이름 설정
	ITensor* prep = preprocess_layer->getOutput(0);

	preprocess_layer->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*prep);

	//auto conv0 = convBlock(network, weightMap, *data, get_width(64, gw), 6, 2, 1, "model.0");
	//assert(conv0);
	//auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	//auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	//auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	//auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
	//auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	//auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	//auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
	//auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
	//auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
	///* ------ yolov5 head ------ */
	//auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

	//auto upsample11 = network->addResize(*conv10->getOutput(0));
	//assert(upsample11);
	//upsample11->setResizeMode(ResizeMode::kNEAREST);
	//upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

	//ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	//auto cat12 = network->addConcatenation(inputTensors12, 2);
	//auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
	//auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

	//auto upsample15 = network->addResize(*conv14->getOutput(0));
	//assert(upsample15);
	//upsample15->setResizeMode(ResizeMode::kNEAREST);
	//upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

	//ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
	//auto cat16 = network->addConcatenation(inputTensors16, 2);

	//auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

	///* ------ detect ------ */
	//IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	//auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
	//ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
	//auto cat19 = network->addConcatenation(inputTensors19, 2);
	//auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
	//IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	//auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
	//ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
	//auto cat22 = network->addConcatenation(inputTensors22, 2);
	//auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
	//IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

	//auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
	//yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	//network->markOutput(*yolo->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

	if (precision_mode == 16) {
		std::cout << "==== precision f16 ====" << std::endl << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}
	else if (precision_mode == 8) {
		std::cout << "==== precision int8 ====" << std::endl << std::endl;
		std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, 0, "../data_calib/", "../Int8_calib_table/yolov5s_int8_calib.table", INPUT_BLOB_NAME);
		config->setInt8Calibrator(calibrator);
	}
	else {
		std::cout << "==== precision f32 ====" << std::endl << std::endl;
	}

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
	char engineFileName[] = "yolov5s";

	char engine_file_path[256];
	sprintf(engine_file_path, "../Engine/%s_%d.engine", engineFileName, precision_mode);

	// 1) engine file 만들기 
	// 강제 만들기 true면 무조건 다시 만들기
	// 강제 만들기 false면, engine 파일 있으면 안만들고 
	//					   engine 파일 없으면 만듬
	bool exist_engine = false;
	if ((access(engine_file_path, 0) != -1)) {
		exist_engine = true;
	}

	if (!((serialize == false)/*Serialize 강제화 값*/ == (exist_engine == true) /*.engine 파일이 있는지 유무*/)) {
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
	std::string img_dir = "../TestDate2/";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
		std::cerr << "[ERROR] Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
	}
	cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);
	cv::Mat ori_img;
	for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
		cv::Mat ori_img = cv::imread(file_names[idx]);
		int ori_w = ori_img.cols;
		int ori_h = ori_img.rows;
		if (ori_h == ori_w) { // 입력이미지가 정사각형일 경우
			cv::Mat img_r(INPUT_H, INPUT_W, CV_8UC3);
			cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 모델 사이즈로 리사이즈
			memcpy(input.data(), img_r.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
		}
		else {
			int new_h, new_w;
			if (ori_w >= ori_h) {
				new_h = (int)(ori_h * ((float)INPUT_W / ori_w));
				new_w = INPUT_W;
			}
			else {
				new_h = INPUT_H;
				new_w = (int)(ori_w * ((float)INPUT_H / ori_h));
			}
			cv::Mat img_r(new_h, new_w, CV_8UC3);
			cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 정합성 일치

			int tb = (int)((INPUT_H - new_h) / 2);
			int bb = ((new_h % 2) == 1) ? tb + 1 : tb;
			int lb = (int)((INPUT_W - new_w) / 2);
			int rb = ((new_w % 2) == 1) ? lb + 1 : lb;
			cv::Mat img_p(INPUT_H, INPUT_W, CV_8UC3);
			cv::copyMakeBorder(img_r, img_p, tb, bb, lb, rb, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			memcpy(input.data(), img_p.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
		}
	}

	std::ofstream ofs("../Validation_py/trt_1", std::ios::binary);
	if (ofs.is_open())
		ofs.write((const char*)input.data(), input.size() * sizeof(uint8_t));
	ofs.close();
	std::exit(0);

	std::cout << "===== input load done =====" << std::endl << std::endl;

	uint64_t dur_time = 0;
	uint64_t iter_count = 100;

	// CUDA 스트림 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	//속도 측정에서 첫 1회 연산 제외하기 위한 계산
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

	// 6) 결과 출력
	std::cout << "==================================================" << std::endl;
	std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << std::endl;
	std::cout << iter_count << " th Iteration, Total dur time : " << dur_time << " [milliseconds]" << std::endl;
	int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	std::cout << "Index : " << max_index << ", Feature_value : " << outputs[max_index] << std::endl;
	std::cout << "Class Name : " << class_names[max_index] << std::endl;
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

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	int p = ksize / 3;
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });
	conv1->setPaddingNd(DimsHW{ p, p });
	conv1->setNbGroups(g);
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

	// silu = x * sigmoid
	auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
	assert(sig);
	auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
	assert(ew);
	return ew;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}


ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname) {
	ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
	ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 4);
	auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
	return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
	auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
	auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
	if (shortcut && c1 == c2) {
		auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
		return ew;
	}
	return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	int c_ = (int)((float)c2 * e);
	auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
	auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
	ITensor *y1 = cv1->getOutput(0);
	for (int i = 0; i < n; i++) {
		auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
		y1 = b->getOutput(0);
	}
	auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

	ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);

	IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
	auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
	lr->setAlpha(0.1);

	auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
	return cv4;
}

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
	int c_ = (int)((float)c2 * e);
	auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
	auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
	ITensor *y1 = cv1->getOutput(0);
	for (int i = 0; i < n; i++) {
		auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
		y1 = b->getOutput(0);
	}

	ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);

	auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
	return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
	int c_ = c1 / 2;
	auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

	auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
	pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
	pool1->setStrideNd(DimsHW{ 1, 1 });
	auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
	pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
	pool2->setStrideNd(DimsHW{ 1, 1 });
	auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
	pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
	pool3->setStrideNd(DimsHW{ 1, 1 });

	ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 4);

	auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
	return cv2;
}

ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname) {
	int c_ = c1 / 2;
	auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

	auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
	pool1->setPaddingNd(DimsHW{ k / 2, k / 2 });
	pool1->setStrideNd(DimsHW{ 1, 1 });
	auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
	pool2->setPaddingNd(DimsHW{ k / 2, k / 2 });
	pool2->setStrideNd(DimsHW{ 1, 1 });
	auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
	pool3->setPaddingNd(DimsHW{ k / 2, k / 2 });
	pool3->setStrideNd(DimsHW{ 1, 1 });
	ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 4);
	auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
	return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
	std::vector<std::vector<float>> anchors;
	Weights wts = weightMap[lname + ".anchor_grid"];
	int anchor_len = CHECK_COUNT * 2;
	for (int i = 0; i < wts.count / anchor_len; i++) {
		auto *p = (const float*)wts.values + i * anchor_len;
		std::vector<float> anchor(p, p + anchor_len);
		anchors.push_back(anchor);
	}
	return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
	auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
	auto anchors = getAnchors(weightMap, lname);
	PluginField plugin_fields[2];
	int netinfo[4] = { CLASS_NUM, INPUT_W, INPUT_H, MAX_OUTPUT_BBOX_COUNT };
	plugin_fields[0].data = netinfo;
	plugin_fields[0].length = 4;
	plugin_fields[0].name = "netinfo";
	plugin_fields[0].type = PluginFieldType::kFLOAT32;
	int scale = 8;
	std::vector<YoloKernel> kernels;
	for (size_t i = 0; i < anchors.size(); i++) {
		YoloKernel kernel;
		kernel.width = INPUT_W / scale;
		kernel.height = INPUT_H / scale;
		memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
		kernels.push_back(kernel);
		scale *= 2;
	}
	plugin_fields[1].data = &kernels[0];
	plugin_fields[1].length = kernels.size();
	plugin_fields[1].name = "kernels";
	plugin_fields[1].type = PluginFieldType::kFLOAT32;
	PluginFieldCollection plugin_data;
	plugin_data.nbFields = 2;
	plugin_data.fields = plugin_fields;
	IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
	std::vector<ITensor*> input_tensors;
	for (auto det : dets) {
		input_tensors.push_back(det->getOutput(0));
	}
	auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
	return yolo;
}