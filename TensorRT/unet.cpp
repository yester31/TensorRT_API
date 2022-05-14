#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 512;
static const int INPUT_W = INPUT_H;
static const int INPUT_C = 3;
static const int class_count = 2;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W * class_count;
static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

ILayer* outConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname);
ILayer* up(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname);
ILayer* down(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int p, std::string lname);
ILayer* doubleConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch);
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName) {
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::map<std::string, Weights> weightMap = loadWeights("../Unet_py/unet.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	preprocess_layer->setName("[preprocess_layer]"); // layer 이름 설정

	// build network
	auto x1 = doubleConv(network, weightMap, *preprocess_layer->getOutput(0), 64, 3, "inc", 64);
	auto x2 = down(network, weightMap, *x1->getOutput(0), 128, 1, "down1");
	auto x3 = down(network, weightMap, *x2->getOutput(0), 256, 1, "down2");
	auto x4 = down(network, weightMap, *x3->getOutput(0), 512, 1, "down3");
	auto x5 = down(network, weightMap, *x4->getOutput(0), 512, 1, "down4");
	ILayer* x6 = up(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up1");
	ILayer* x7 = up(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up2");
	ILayer* x8 = up(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up3");
	ILayer* x9 = up(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up4");
	ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), OUTPUT_SIZE, "outc");
	x10->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*x10->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1ULL << 29);  // 512MB
	//std::cout << 28 * (1 << 23) << std::endl << std::endl; // 224MB
	//std::cout << 28 * (1 << 24) << std::endl << std::endl; // 448MB

	if (precision_mode == 16) {
		std::cout << "==== precision f16 ====" << std::endl << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}
	else if (precision_mode == 8) {
		std::cout << "==== precision int8 ====" << std::endl << std::endl;
		std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		const char* data_calib_path = "../data_calib/";
		Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_W, INPUT_H, 1, data_calib_path, "../Int8_calib_table/unet_int8_calib.table", INPUT_BLOB_NAME);
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
		//return -1;
	}
	p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
	std::cout << "==== model selialize done ====" << std::endl << std::endl;

	engine->destroy();
	network->destroy();
	p.close();
	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
}

int main()
{
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	bool serialize = false;			// Serialize 강제화 시키기(true 엔진 파일 생성)
	char engineFileName[] = "unet";
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
	
	// CPU에서 입력과 출력으로 사용할 메모리 공간할당
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> outputs(OUTPUT_SIZE);

	// 4) 입력으로 사용할 이미지 준비하기 (resize & letterbox padding) openCV 사용
	std::string img_dir = "../Unet_py/data";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
		std::cerr << "[ERROR] Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
	}
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
			//cv::resize(ori_img, img_r, img_r.size(), cv::INTER_AREA); // INTER_AREA 알고리즘 에러 (c 와 python 결과 불일치)
			cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 정합성 일치

			int tb = (int)((INPUT_H - new_h) / 2);
			int bb = ((new_h % 2) == 1) ? tb + 1 : tb;
			int lb = (int)((INPUT_W - new_w) / 2);
			int rb = ((new_w % 2) == 1) ? lb + 1 : lb;
			cv::Mat img_p(INPUT_H, INPUT_W, CV_8UC3);
			cv::copyMakeBorder(img_r, img_p, tb, bb, lb, rb, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
			memcpy(input.data(), img_p.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
		}
	}
	//std::ofstream ofs("../Validation_py/trt_1", std::ios::binary);
	//if (ofs.is_open())
	//	ofs.write((const char*)input.data(), input.size() * sizeof(uint8_t));
	//ofs.close();
	//std::exit(0);

	std::cout << "===== input load done =====" << std::endl << std::endl;

	uint64_t dur_time = 0;
	uint64_t iter_count = 100;

	// CUDA 스트림 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	//warm-up
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
	context->enqueue(maxBatchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	//tofile(outputs, "../Validation_py/trt");
	//std::exit(0);

	// 5) Inference 수행  
	for (int i = 0; i < iter_count; i++) {
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
	std::cout << iter_count << " th Iteration" << std::endl;
	std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
	std::cout << "Avg duration time with data transfer : " << dur_time / iter_count << " [milliseconds]" << std::endl;
	std::cout << "FPS : " << 1000.f / (dur_time / iter_count) << " [frame/sec]" << std::endl;
	std::cout << "===== TensorRT Model Calculate done =====" << std::endl;
	std::cout << "==================================================" << std::endl;

	// 이미지 출력 로직
	std::vector<uint8_t> show_img(maxBatchSize * INPUT_C * INPUT_H * INPUT_W);
	std::vector<float> prob(maxBatchSize * INPUT_H * INPUT_W);
	std::vector<uint8_t> index_c(maxBatchSize * INPUT_H * INPUT_W);
	std::vector<std::vector<unsigned char>> color = { {0,0,0}, {255,255,255} }; // class 수 만큼 색 준비

	for (int midx = 0; midx < prob.size(); midx++) {
		float sum = 0.f;
		float max = 0.f;
		int class_index;
		for (int i = 0; i < class_count; i++)
		{
			int ⁠g_idx_i = midx + i * prob.size();
			float z = exp(outputs[⁠g_idx_i]);
			sum += z;
			if (max < z) {
				class_index = i;
				max = z;
			}
		}
		prob[midx] = max / sum;
		index_c[midx] = class_index;
		show_img[midx * 3]		= color[class_index][0];
		show_img[midx * 3 + 1]	= color[class_index][1];
		show_img[midx * 3 + 2]	= color[class_index][2];
	}

	cv::Mat frame = cv::Mat(INPUT_H , INPUT_W, CV_8UC3, show_img.data());
	cv::imshow("result", frame);
	cv::waitKey(0);

	// Release...
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) 
{
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
	return scale_1;
}


ILayer* doubleConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) 
{
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".double_conv.0.weight"], weightMap[lname + ".double_conv.0.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 1E-05);
	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".double_conv.3.weight"], weightMap[lname + ".double_conv.3.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 1E-05);
	IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
	return relu2;
}


ILayer* down(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int p, std::string lname) 
{
	IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
	ILayer* dcov1 = doubleConv(network, weightMap, *pool1->getOutput(0), outch, 3, lname + ".maxpool_conv.1", outch);
	return dcov1;
}

ILayer* up(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) 
{
	float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
	for (int i = 0; i < resize * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	ITensor* upsampleTensor;
	if (false) {
		Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
		Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
		IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
		deconv1->setStrideNd(DimsHW{ 2, 2 });
		deconv1->setNbGroups(resize);
		weightMap["deconvwts." + lname] = deconvwts1;
		upsampleTensor = deconv1->getOutput(0);
	}
	else {
		IResizeLayer* resize = network->addResize(input1);
		std::vector<float> scale{ 1.f, 2, 2 };
		resize->setScales(scale.data(), scale.size());
		resize->setAlignCorners(true);
		resize->setResizeMode(ResizeMode::kLINEAR);
		upsampleTensor = resize->getOutput(0);
	}

	int diffx = input2.getDimensions().d[1] - upsampleTensor->getDimensions().d[1];
	int diffy = input2.getDimensions().d[2] - upsampleTensor->getDimensions().d[2];

	ILayer* pad1 = network->addPaddingNd(*upsampleTensor, DimsHW{ diffx / 2, diffy / 2 }, DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);

	if (midch == 64) {
		ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
		return dcov1;
	}
	else {
		int midch1 = outch / 2;
		ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch1, 3, lname + ".conv", outch);
		return dcov1;
	}
}

ILayer* outConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) 
{
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, class_count, DimsHW{ 1, 1 }, weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 0, 0 });
	conv1->setNbGroups(1);
	conv1->setName("[last_layer]"); // layer 이름 설정
	return conv1;
}