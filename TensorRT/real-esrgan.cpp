#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "postprocess.hpp"	// postprocess plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 220;
static const int INPUT_W = 220;
static const int INPUT_C = 3;
static const int OUT_SCALE = 2;
static const int OUTPUT_SIZE = INPUT_C * INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE;
static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

ITensor* residualDenseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname);
ITensor* RRDB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname);

// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
    std::cout << "==== model build start ====" << std::endl << std::endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);
    std::map<std::string, Weights> weightMap;
    if (OUT_SCALE == 2) {
        weightMap = loadWeights("../Real-ESRGAN_py/RealESRGAN_x2plus.wts");
    }
    else {
        weightMap = loadWeights("../Real-ESRGAN_py/real-esrgan.wts");
    }
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
    assert(data);

    Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 
    IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
    IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);
    IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);
    preprocess_layer->setName("preprocess_layer");
    ITensor* prep = preprocess_layer->getOutput(0);

    if (OUT_SCALE == 2) {
        // Pixel unshuffle.
        int h = INPUT_H / OUT_SCALE;
        int w = INPUT_W / OUT_SCALE;
        auto attn_shuffle = network->addShuffle(*prep);

        Dims shape_dims;
        std::vector<int> reshape_dims = { INPUT_C, h, OUT_SCALE, w, OUT_SCALE };
        shape_dims.nbDims = (int)reshape_dims.size();
        memcpy(shape_dims.d, reshape_dims.data(), reshape_dims.size() * sizeof(int));
        attn_shuffle->setReshapeDimensions(shape_dims);

        std::vector<int> trans_dims{ 0, 2, 4, 1, 3 };
        Permutation f_trans_dims; memcpy(f_trans_dims.order, trans_dims.data(), trans_dims.size() * sizeof(int));
        attn_shuffle->setSecondTranspose(f_trans_dims);

        auto attn_shuffle2 = network->addShuffle(*attn_shuffle->getOutput(0));
        attn_shuffle2->setReshapeDimensions(Dims3{ INPUT_C * OUT_SCALE * OUT_SCALE, h, w});
        prep = attn_shuffle2->getOutput(0);
    }

    // conv_first
    IConvolutionLayer* conv_first = network->addConvolutionNd(*prep, 64, DimsHW{ 3, 3 }, weightMap["conv_first.weight"], weightMap["conv_first.bias"]);
    conv_first->setStrideNd(DimsHW{ 1, 1 });
    conv_first->setPaddingNd(DimsHW{ 1, 1 });
    conv_first->setName("conv_first");
    ITensor* feat = conv_first->getOutput(0);

    // conv_body
    ITensor* body_feat = RRDB(network, weightMap, feat, "body.0");
    for (int idx = 1; idx < 23; idx++) {
        body_feat = RRDB(network, weightMap, body_feat, "body." + std::to_string(idx));
    }

    IConvolutionLayer* conv_body = network->addConvolutionNd(*body_feat, 64, DimsHW{ 3, 3 }, weightMap["conv_body.weight"], weightMap["conv_body.bias"]);
    conv_body->setStrideNd(DimsHW{ 1, 1 });
    conv_body->setPaddingNd(DimsHW{ 1, 1 });
    IElementWiseLayer* ew1 = network->addElementWise(*feat, *conv_body->getOutput(0), ElementWiseOperation::kSUM);
    feat = ew1->getOutput(0);

    //upsample
    IResizeLayer* interpolate_nearest = network->addResize(*feat);
    //Dims3 sclaes1 = { feat->getDimensions().d[0], feat->getDimensions().d[1]*2, feat->getDimensions().d[2]*2 };
    //interpolate_nearest->setResizeMode(ResizeMode::kNEAREST);
    //interpolate_nearest->setOutputDimensions(sclaes1);
    float sclaes1[] = { 1, 2, 2 };
    interpolate_nearest->setScales(sclaes1, 3);
    interpolate_nearest->setResizeMode(ResizeMode::kNEAREST);

    IConvolutionLayer* conv_up1 = network->addConvolutionNd(*interpolate_nearest->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_up1.weight"], weightMap["conv_up1.bias"]);
    conv_up1->setStrideNd(DimsHW{ 1, 1 });
    conv_up1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_up1->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_1->setAlpha(0.2);

    IResizeLayer* interpolate_nearest2 = network->addResize(*leaky_relu_1->getOutput(0));
    float sclaes2[] = { 1, 2, 2 };
    interpolate_nearest2->setScales(sclaes2, 3);
    interpolate_nearest2->setResizeMode(ResizeMode::kNEAREST);
    IConvolutionLayer* conv_up2 = network->addConvolutionNd(*interpolate_nearest2->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_up2.weight"], weightMap["conv_up2.bias"]);
    conv_up2->setStrideNd(DimsHW{ 1, 1 });
    conv_up2->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_up2->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_2->setAlpha(0.2);

    IConvolutionLayer* conv_hr = network->addConvolutionNd(*leaky_relu_2->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_hr.weight"], weightMap["conv_hr.bias"]);
    conv_hr->setStrideNd(DimsHW{ 1, 1 });
    conv_hr->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_hr = network->addActivation(*conv_hr->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_hr->setAlpha(0.2);
    IConvolutionLayer* conv_last = network->addConvolutionNd(*leaky_relu_hr->getOutput(0), 3, DimsHW{ 3, 3 }, weightMap["conv_last.weight"], weightMap["conv_last.bias"]);
    conv_last->setStrideNd(DimsHW{ 1, 1 });
    conv_last->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* out = conv_last->getOutput(0);

    //postprocess (RGB -> BGR, NCHW->NHWC, *255, ROUND, uint8)
    Postprocess postprocess{ maxBatchSize, out->getDimensions().d[0], out->getDimensions().d[1], out->getDimensions().d[2] };
    IPluginCreator* postprocess_creator = getPluginRegistry()->getPluginCreator("postprocess", "1");
    IPluginV2 *postprocess_plugin = postprocess_creator->createPlugin("postprocess_plugin", (PluginFieldCollection*)&postprocess);
    IPluginV2Layer* postprocess_layer = network->addPluginV2(&out, 1, *postprocess_plugin);
    postprocess_layer->setName("postprocess_layer");

    ITensor* final_tensor = postprocess_layer->getOutput(0);
    //show_dims(final_tensor);
    final_tensor->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*final_tensor);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 29);  // 512MB

    if (precision_mode == 16) {
        std::cout << "==== precision f16 ====" << std::endl << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == 8) {
        //std::cout << "==== precision int8 ====" << std::endl << std::endl;
        //std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        //assert(builder->platformHasFastInt8());
        //config->setFlag(BuilderFlag::kINT8);
        //Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_W, INPUT_H, 0, "../data_calib/", "../Int8_calib_table/real-esrgan_int8_calib.table", INPUT_BLOB_NAME);
        //config->setInt8Calibrator(calibrator);
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
    unsigned int maxBatchSize = 1;			// batch size 
    bool serialize = true;					// TensorRT Model Serialize flag(true : generate engine, false : if no engine file, generate engine )
    char engineFileName[] = "real-esrgan";	// model name

    char engine_file_path[256];
    sprintf(engine_file_path, "../Engine/%s_%d.engine", engineFileName, precision_mode);

    // checking engine file existence
    bool exist_engine = false;
    if ((access(engine_file_path, 0) != -1)) {
        exist_engine = true;
    }

    // 1) Generation engine file (decide whether to create a new engine with serialize and exist_engine variable)
    if (!((serialize == false)/*Serialize flag*/ && (exist_engine == true) /*engine existence flag*/)) {
        std::cout << "===== Create Engine file =====" << std::endl << std::endl;
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();
        createEngine(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path); // generation TensorRT Model
        builder->destroy();
        config->destroy();
        std::cout << "===== Create Engine file =====" << std::endl << std::endl;
    }

    // 2) Load engine file 
    char *trtModelStream{ nullptr };
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

    // 3) Engine file deserialize
    std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;
    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // Allocating memory space for inputs and outputs on the GPU
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(uint8_t)));

    // 4) Prepare image data for inputs
    std::string img_dir = "../Real-ESRGAN_py/data/";
    std::vector<std::string> file_names;
    if (SearchFile(img_dir.c_str(), file_names) < 0) { // search files
        std::cerr << "[ERROR] Data search error" << std::endl;
    }
    else {
        std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
    }

    cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    cv::Mat ori_img;
    std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<uint8_t> outputs(OUTPUT_SIZE);
    for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
        cv::Mat ori_img = cv::imread(file_names[idx]);
        cv::resize(ori_img, img, img.size()); // resize image to input size
        memcpy(input.data(), img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    }
    std::cout << "===== input load done =====" << std::endl << std::endl;

    uint64_t dur_time = 0;
    uint64_t iter_count = 10;

    // Generate CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // warm-up
    std::cout << "===== warm-up begin =====" << std::endl << std::endl;
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    context->enqueue(maxBatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    std::cout << "===== warm-up done =====" << std::endl << std::endl;

    // 5) Inference  
    for (int i = 0; i < iter_count; i++) {
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        context->enqueue(maxBatchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
    }

    // 6) Print results
    std::cout << "==================================================" << std::endl;
    std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << std::endl;
    std::cout << iter_count << " th Iteration" << std::endl;
    std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
    std::cout << "Avg duration time with data transfer : " << dur_time / iter_count << " [milliseconds]" << std::endl;
    std::cout << "FPS : " << 1000.f / (dur_time / iter_count) << " [frame/sec]" << std::endl;
    std::cout << "===== TensorRT Model Calculate done =====" << std::endl;
    std::cout << "==================================================" << std::endl;

    cv::Mat frame = cv::Mat(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3, outputs.data());
    cv::imshow("result", frame);
    cv::waitKey(0);
    cv::imwrite("../result.png", frame);
    //tofile(outputs, "../Validation_py/c"); // ouputs data to files

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
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
