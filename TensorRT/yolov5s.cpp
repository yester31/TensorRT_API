#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "yololayer.hpp"	// yololayer plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;


static const int CHECK_COUNT = 3;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int INPUT_C = 3;
static const int CLASS_NUM = 80;
static const int MAX_OUTPUT_BBOX_COUNT = 300;
static const int OUTPUT_SIZE = 6 * MAX_OUTPUT_BBOX_COUNT;  
//static const int OUTPUT_SIZE = 6 * 25200;  
static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

// yolov5s 
static const float  gd = 0.33;
static const float  gw = 0.50;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// COCO dataset class names2
static std::vector<std::string> COCO_names2{
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
	"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
	"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
	"sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);
ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname);
ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname);
ITensor* add_YoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, ITensor& input, int grid_stride);

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
	float nw, nh, nx,ny;
	float l, r, t, b;
	float ratio = std::min(((float)INPUT_W / img.cols), ((float)INPUT_H / img.rows));

	int new_h = (int)(round(img.rows * ratio));
	int new_w = (int)(round(img.cols * ratio));

	//int dh = (INPUT_H - new_h) % 32;
	//int dw = (INPUT_W - new_w) % 32;
	int dh = (INPUT_H - new_h);
	int dw = (INPUT_W - new_w);

	int tb = (int)round(((float)dh / 2) - 0.1);
	int bb = (int)round(((float)dh / 2) + 0.1);
	int lb = (int)round(((float)dw / 2) - 0.1);
	int rb = (int)round(((float)dw / 2) + 0.1);

	l = (bbox[0] - bbox[2] / 2.f - lb) / ratio;
	r = (bbox[0] + bbox[2] / 2.f - rb) / ratio;
	t = (bbox[1] - bbox[3] / 2.f - tb) / ratio;
	b = (bbox[1] + bbox[3] / 2.f - bb) / ratio;

	nx = l;
	ny = t;
	nw = r - l;
	nh = b - t;

	return cv::Rect(round(nx), round(ny), round(nw), round(nh));
}

float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

struct alignas(float) Detection {
	//center_x center_y w h
	float bbox[4];
	float conf;  // bbox_conf * cls_conf
	float class_id;
};

void nms(std::vector<Detection>& res, float *output, float conf_thresh = 0.25, float nms_thresh = 0.45) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	
	for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}

	for (auto it = m.begin(); it != m.end(); it++) {
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

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
	preprocess_layer->setName("preprocess_layer"); // layer 이름 설정
	//preprocess_layer->getOutput(0)->setName(OUTPUT_BLOB_NAME);// 출력값 Tensor 이름을 출력 이름으로 설정 
	//network->markOutput(*preprocess_layer->getOutput(0));// preprocess_layer의 출력값을 모델 Output으로 설정
	ITensor* prep = preprocess_layer->getOutput(0);

	auto conv0 = convBlock(network, weightMap,*prep, get_width(64, gw), 6, 2, 1, "model.0");
	assert(conv0);

	auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
	auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
	auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
	auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
	/* ------ yolov5 head ------ */
	auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

	auto upsample11 = network->addResize(*conv10->getOutput(0));
	assert(upsample11);
	upsample11->setResizeMode(ResizeMode::kNEAREST);
	upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

	ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	auto cat12 = network->addConcatenation(inputTensors12, 2);
	auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
	auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

	auto upsample15 = network->addResize(*conv14->getOutput(0));
	assert(upsample15);
	upsample15->setResizeMode(ResizeMode::kNEAREST);
	upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

	ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
	auto cat16 = network->addConcatenation(inputTensors16, 2);
	auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

	auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
	ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
	auto cat19 = network->addConcatenation(inputTensors19, 2);
	auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");

	auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
	ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
	auto cat22 = network->addConcatenation(inputTensors22, 2);
	auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

	/* ------ detect ------ */
	IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

	auto yolo_t0 = add_YoLoLayer(network, weightMap, "model.24.anchor_grid0", *(det0->getOutput(0)), 8);
	auto yolo_t1 = add_YoLoLayer(network, weightMap, "model.24.anchor_grid1", *(det1->getOutput(0)), 16);
	auto yolo_t2 = add_YoLoLayer(network, weightMap, "model.24.anchor_grid2", *(det2->getOutput(0)), 32);

	ITensor* yolo_ts[] = {yolo_t0, yolo_t1, yolo_t2};
	auto concat_layer = network->addConcatenation(yolo_ts, 3);
	//cat->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	//network->markOutput(*cat->getOutput(0));

	auto slice_layer = network->addSlice(*concat_layer->getOutput(0), Dims2(0, 4), Dims2(concat_layer->getOutput(0)->getDimensions().d[0], 1), Dims2(1, 1));
	auto sort_layer = network->addTopK(*slice_layer->getOutput(0), TopKOperation::kMAX, MAX_OUTPUT_BBOX_COUNT, 1 << 0);
	auto shuffle_layer = network->addShuffle(*sort_layer->getOutput(1));
	Dims dims_shape; dims_shape.nbDims = 1; dims_shape.d[0] = MAX_OUTPUT_BBOX_COUNT;
	shuffle_layer->setReshapeDimensions(dims_shape);
	auto gather_layer = network->addGather(*concat_layer->getOutput(0), *shuffle_layer->getOutput(0), 0);
	ITensor* final_out = gather_layer->getOutput(0);
	final_out->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*final_out);
	
	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1ULL << 31);  // 2,048MB

	if (precision_mode == 16) {
		std::cout << "==== precision f16 ====" << std::endl << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}
	else if (precision_mode == 8) {
		std::cout << "==== precision int8 ====" << std::endl << std::endl;
		std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, 2, "../data_calib/", "../Int8_calib_table/yolov5s_int8_calib.table", INPUT_BLOB_NAME);
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
	p.close();
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

	if (!((serialize == false)/*Serialize 강제화 값*/ && (exist_engine == true) /*.engine 파일이 있는지 유무*/)) {
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
	std::string img_dir = "../yolov5s_py/data/";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
		std::cerr << "[ERROR] Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
	}
	cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> outputs(maxBatchSize * OUTPUT_SIZE);
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
			float ratio = std::min(((float)INPUT_W / ori_w), ((float)INPUT_H / ori_h));

			int	new_h = (int)(round(ori_h * ratio));
			int	new_w = (int)(round(ori_w * ratio));

			cv::Mat img_r(new_h, new_w, CV_8UC3);
			cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR); // 정합성 일치
			
			//int dh = (INPUT_H - new_h) % 32;
			//int dw = (INPUT_W - new_w) % 32;
			int dh = (INPUT_H - new_h);
			int dw = (INPUT_W - new_w);

			int tb = (int)round(((float)dh / 2) - 0.1);
			int bb = (int)round(((float)dh / 2) + 0.1);
			int lb = (int)round(((float)dw / 2) - 0.1);
			int rb = (int)round(((float)dw / 2) + 0.1);

			cv::Mat img_p((new_h + dh), (new_w + dw), CV_8UC3);
			cv::copyMakeBorder(img_r, img_p, tb, bb, lb, rb, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			input.resize((new_h + dh) * (new_w + dw) * INPUT_C);
			memcpy(input.data() + idx * ((new_h + dh) * (new_w + dw) * INPUT_C), img_p.data, (new_h + dh) * (new_w + dw) * INPUT_C);
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

	//속도 측정에서 첫 1회 연산 제외하기 위한 계산
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
	context->enqueue(maxBatchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	//tofile(outputs);
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

	if (true) {
		std::vector<std::vector<Detection>> batch_res(maxBatchSize);
		for (int b = 0; b < maxBatchSize; b++) {
			auto& res = batch_res[b];
			nms(res, &outputs[b * OUTPUT_SIZE]);
		}
		for (int b = 0; b < maxBatchSize; b++) {
			auto& res = batch_res[b];
			cv::Mat img = cv::imread(file_names[b]);
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r = get_rect(img, res[j].bbox);
				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(img, COCO_names2[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			}
			cv::imshow(engineFileName, img);
			cv::waitKey(0);
		}
	}

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

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
	auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
	auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
	if (shortcut && c1 == c2) {
		auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
		return ew;
	}
	return cv2;
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

ITensor* add_YoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, ITensor& input, int grid_stride)
{
	IShuffleLayer* shuffle_layer = network->addShuffle(input);
	shuffle_layer->setReshapeDimensions(Dims4(3, CLASS_NUM + 5, input.getDimensions().d[1], input.getDimensions().d[2]));
	std::vector<int> trans_dims{ 0, 2, 3, 1 };
	Permutation second_trans_dims; memcpy(second_trans_dims.order, trans_dims.data(), trans_dims.size() * sizeof(int));
	shuffle_layer->setSecondTranspose(second_trans_dims);
	IActivationLayer* sigmoid_layer = network->addActivation(*shuffle_layer->getOutput(0), ActivationType::kSIGMOID);
	ITensor* anchor_grid = network->addConstant(Dims2(3, 2), weightMap[lname])->getOutput(0);
	Yololayer yololayer_vs{ 3, input.getDimensions().d[1], input.getDimensions().d[2], CLASS_NUM, grid_stride };
	IPluginCreator* creator0 = getPluginRegistry()->getPluginCreator("yololayer", "1");
	IPluginV2 *plugin0 = creator0->createPlugin("yololayer_plugin", (PluginFieldCollection*)&yololayer_vs);
	std::vector<ITensor*> datas{ sigmoid_layer->getOutput(0), anchor_grid };
	IPluginV2Layer* plugin_layer0 = network->addPluginV2(datas.data(), (int)datas.size(), *plugin0);
	return plugin_layer0->getOutput(0);
}
