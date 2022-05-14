// 2021-8-19 by YH PARK 
// custom plugin
// preprocess(NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	

using namespace nvinfer1;
sample::Logger gLogger;

void main() 
{
	std::cout << "===== custom plugin example start =====" << std::endl;
	char engineFileName[] = "../Engine/plugin_test.engine";

	// 0. 이미지들의 저장 경로 불러오기
	std::string img_dir = "../TestDate/";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) {
		std::cerr << "Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : "<< file_names.size() << std::endl;
	}

	// 1. 이미지 데이터 로드
	int batch_size{ 1 };
	int input_width{ 224 };
	int input_height{ 224 };
	int input_channel{ 3 };
	const char* INPUT_NAME = "inputs";
	const char* OUTPUT_NAME = "outputs";
	cv::Mat img(input_height, input_width, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(batch_size * input_height * input_width * input_channel);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> output(batch_size* input_channel * input_height * input_width);		// 출력이 담길 컨테이너 변수 생성
	
	for (int idx = 0; idx < file_names.size(); idx++) {
		cv::Mat ori_img = cv::imread(file_names[idx]);
		//cv::resize(ori_img, img, img.size()); // input size로 리사이즈
		memcpy(input.data(), ori_img.data, batch_size * input_height * input_width * input_channel * sizeof(uint8_t));
	}
	std::cout << "===== input load done =====" << std::endl;

	//==========================================================================================

	std::cout << "===== Create TensorRT Model =====" << std::endl; // TensorRT 모델 만들기 시작
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	
	// 네트워크 구조를 만들기 위해 네트워크 객체 생성
	INetworkDefinition* network = builder->createNetworkV2(0U);
	
	// 입력(Input) 레이어 생성
	ITensor* input_tensor = network->addInput(INPUT_NAME, nvinfer1::DataType::kFLOAT, Dims3{input_height, input_width, input_channel }); // [N,C,H,W]
	
	// Custom(preprocess) plugin 사용하기
	// Custom(preprocess) plugin에서 사용할 구조체 객체 생성
	Preprocess preprocess{batch_size, input_channel, input_height, input_width, 0};
	// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
	// Custom(preprocess) plugin 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);
	// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&input_tensor, 1, *preprocess_plugin);
	preprocess_layer->setName("preprocess_layer"); // layer 이름 설정
	preprocess_layer->getOutput(0)->setName(OUTPUT_NAME);// 출력값 Tensor 이름을 출력 이름으로 설정 
	network->markOutput(*preprocess_layer->getOutput(0));// preprocess_layer의 출력값을 모델 Output으로 설정

	builder->setMaxBatchSize(maxBatchSize); // 모델의 배치 사이즈 설정
	config->setMaxWorkspaceSize(1ULL << 26); // 64MB, 엔진 생성을 위해 사용할 메모리 공간 설정

	std::cout << "Building engine, please wait for a while..." << std::endl;
	IHostMemory* engine0 = builder->buildSerializedNetwork(*network, *config); // 엔진 생성(빌드)
	std::cout << "==== model build done ====" << std::endl << std::endl;

	std::cout << "==== model selialize start ====" << std::endl << std::endl;
	std::ofstream p(engineFileName, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
	}
	p.write(reinterpret_cast<const char*>(engine0->data()), engine0->size());
	std::cout << "==== model selialize done ====" << std::endl << std::endl;
	builder->destroy();
	config->destroy();
	engine0->destroy();
	network->destroy();
	p.close();
	//==========================================================================================

	std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
	char *trtModelStream{ nullptr };// 저장된 스트림을 저장할 변수
	size_t size{ 0 };
	std::cout << "===== Engine file load =====" << std::endl << std::endl;
	std::ifstream file(engineFileName, std::ios::binary);
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
	std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	IExecutionContext* context = engine->createExecutionContext();
	delete[] trtModelStream;
	void* buffers[2]; // 입력값과 출력값을 주고 받기 위해 포인터 변수 생성 

	// 네트워크 생성할때 사용한 입력과 출력의 이름으로 인덱스값 받아 오기 
	const int inputIndex = engine->getBindingIndex(INPUT_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_NAME);

	// GPU에 버퍼 생성(device에 저장 공간 할당)
	CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * input_channel * input_height * input_width * sizeof(uint8_t)));
	CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * input_channel * input_height * input_width * sizeof(float)));

	// Cuda 스트림 객체 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	// GPU로 입력 데이터 전달 (CPU -> GPU)
	CHECK(cudaMemcpyAsync((uint8_t*)buffers[inputIndex], (char*)input.data(), maxBatchSize * input_channel * input_height * input_width * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
	// 배치단위로 비동기로 작업 수행 
	context->enqueue(maxBatchSize, buffers, stream, nullptr);
	// CPU로 출력 데이터 가져오기 (CPU <- GPU)
	CHECK(cudaMemcpyAsync(output.data(), buffers[outputIndex], maxBatchSize * input_channel * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost, stream));
	// 스트림 단위로 동기화 수행
	cudaStreamSynchronize(stream);
	std::cout << "===== TensorRT Model Calculate done =====" << std::endl;
	//==========================================================================================

	tofile(output, "../Validation_py/C_Preproc_Result"); // 결과값 파일로 출력
	//../Validation_py/valide_preproc.py 에서 결과 비교 ㄱㄱ

	// 자원 해제 작업
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	context->destroy();
	engine->destroy();
	runtime->destroy();
}