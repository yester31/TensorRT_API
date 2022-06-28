#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 536;
static const int INPUT_W = 640;
static const int INPUT_C = 3;

static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static ITensor* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
static ITensor* addRepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname, bool rbr_identity);

static ITensor* addSimSPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    //cv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(*input, 256, DimsHW{ 1, 1 }, weightMap[lname + "cv1.conv.weight"], emptywts);
    conv1->setStrideNd(DimsHW{ 1, 1 });

    ITensor* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "cv1.bn", 1e-3);
    auto* nonlinearity0 = network->addActivation(*bn1, ActivationType::kRELU);

    auto pool1 = network->addPoolingNd(*nonlinearity0->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    pool1->setPaddingNd(DimsHW{ 2, 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });

    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    pool2->setPaddingNd(DimsHW{ 2, 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });

    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    pool3->setPaddingNd(DimsHW{ 2, 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputs4[] = { nonlinearity0->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat12 = network->addConcatenation(inputs4, 4);

    //cv2
    IConvolutionLayer* conv2 = network->addConvolutionNd(*cat12->getOutput(0), 512, DimsHW{ 1, 1 }, weightMap[lname + "cv2.conv.weight"], emptywts);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    ITensor* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "cv2.bn", 1e-3);
    auto* nonlinearity1 = network->addActivation(*bn2, ActivationType::kRELU);
    return nonlinearity1->getOutput(0);
}


// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
    std::cout << "==== model build start ====" << std::endl << std::endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov6s_py/yolov6s.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
    assert(data);

    float ratio = std::min(640 / INPUT_H, 640 / INPUT_W);
    int new_unpad_h = ratio * INPUT_H + 0.5f;
    int new_unpad_w = ratio * INPUT_W + 0.5f;

    // Compute padding
    int dh = int(640 - new_unpad_h) % 32;
    int dw = int(640 - new_unpad_w) % 32;

    int P = new_unpad_h + dh;
    int Q = new_unpad_w + dw;

    // divide padding into 2 sides
    int top = round(dh / 2 - 0.1);
    int	bottom = round(dh / 2 + 0.1);
    int left = round(dw / 2 - 0.1);
    int right = round(dw / 2 + 0.1);

    Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 3};// Custom(preprocess) plugin 
    preprocess.P = P;
    preprocess.Q = Q;
    preprocess.P0 = new_unpad_h;
    preprocess.Q0 = new_unpad_w;
    preprocess.pt = top;
    preprocess.pb = bottom;
    preprocess.pl = left;
    preprocess.pr = right;
    IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
    IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);
    IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);
    preprocess_layer->setName("preprocess_layer");
    ITensor* prep = preprocess_layer->getOutput(0);

    // backbone
    // stem
    ITensor* stem = addRepVGGBlock(network, weightMap, prep, 32, "backbone.stem.", false);
    // ERBlock_2
    ITensor* ERBlock_2_0 = addRepVGGBlock(network, weightMap, stem, 64, "backbone.ERBlock_2.0.", false);
    ITensor* ERBlock_2_1 = addRepVGGBlock(network, weightMap, ERBlock_2_0, 64, "backbone.ERBlock_2.1.conv1.", true);
    ITensor* ERBlock_2_2 = addRepVGGBlock(network, weightMap, ERBlock_2_1, 64, "backbone.ERBlock_2.1.block.0.", true);
    // ERBlock_3
    ITensor* ERBlock_3_0 = addRepVGGBlock(network, weightMap, ERBlock_2_2, 128, "backbone.ERBlock_3.0.", false);
    ITensor* ERBlock_3_1 = addRepVGGBlock(network, weightMap, ERBlock_3_0, 128, "backbone.ERBlock_3.1.conv1.", true);
    ITensor* ERBlock_3_2 = addRepVGGBlock(network, weightMap, ERBlock_3_1, 128, "backbone.ERBlock_3.1.block.0.", true);
    ITensor* ERBlock_3_3 = addRepVGGBlock(network, weightMap, ERBlock_3_2, 128, "backbone.ERBlock_3.1.block.1.", true);
    ITensor* ERBlock_3_4 = addRepVGGBlock(network, weightMap, ERBlock_3_3, 128, "backbone.ERBlock_3.1.block.2.", true);
    // ERBlock_4
    ITensor* ERBlock_4_0 = addRepVGGBlock(network, weightMap, ERBlock_3_4, 256, "backbone.ERBlock_4.0.", false);
    ITensor* ERBlock_4_1 = addRepVGGBlock(network, weightMap, ERBlock_4_0, 256, "backbone.ERBlock_4.1.conv1.", true);
    ITensor* ERBlock_4_2 = addRepVGGBlock(network, weightMap, ERBlock_4_1, 256, "backbone.ERBlock_4.1.block.0.", true);
    ITensor* ERBlock_4_3 = addRepVGGBlock(network, weightMap, ERBlock_4_2, 256, "backbone.ERBlock_4.1.block.1.", true);
    ITensor* ERBlock_4_4 = addRepVGGBlock(network, weightMap, ERBlock_4_3, 256, "backbone.ERBlock_4.1.block.2.", true);
    ITensor* ERBlock_4_5 = addRepVGGBlock(network, weightMap, ERBlock_4_4, 256, "backbone.ERBlock_4.1.block.3.", true);
    ITensor* ERBlock_4_6 = addRepVGGBlock(network, weightMap, ERBlock_4_5, 256, "backbone.ERBlock_4.1.block.4.", true);
    // ERBlock_5
    ITensor* ERBlock_5_0 = addRepVGGBlock(network, weightMap, ERBlock_4_6, 512, "backbone.ERBlock_5.0.", false);
    ITensor* ERBlock_5_1 = addRepVGGBlock(network, weightMap, ERBlock_5_0, 512, "backbone.ERBlock_5.1.conv1.", true);
    ITensor* ERBlock_5_2 = addRepVGGBlock(network, weightMap, ERBlock_5_1, 512, "backbone.ERBlock_5.1.block.0.", true);
    ITensor* ERBlock_5_3 = addSimSPPF(network, weightMap, ERBlock_5_2, "backbone.ERBlock_5.2.");
    // backbone

    // neck
    // neck

    // detect
    // detect

    ITensor* final_tensor = ERBlock_5_3;
    show_dims(final_tensor);
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
    unsigned int maxBatchSize = 1;          // batch size 
    bool serialize = true;                 // TensorRT Model Serialize flag(true : generate engine, false : if no engine file, generate engine )
    char engineFileName[] = "yolov6";       // model name
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
    int OUTPUT_SIZE = 1 * 512 * 17 * 20;
    //int OUTPUT_SIZE = 1 * 256 * 34 * 40;
    //int OUTPUT_SIZE = 1 * 128 * 68 * 80;
    //int OUTPUT_SIZE = 1 * 64 * 136 * 160;
    //int OUTPUT_SIZE = 1 * 32 * 272 * 320;
    //int OUTPUT_SIZE = 1 * 3 * 544 * 640;
    std::vector<float> outputs(OUTPUT_SIZE);
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

    // 4) Prepare image data for inputs
    std::string img_dir = "../yolov6s_py/data/";
    std::vector<std::string> file_names;
    if (SearchFile(img_dir.c_str(), file_names) < 0) { // search files
        std::cerr << "[ERROR] Data search error" << std::endl;
    }
    else {
        std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
    }

    cv::Mat ori_img;
    std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
        cv::Mat ori_img = cv::imread(file_names[idx]);
        memcpy(input.data(), ori_img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
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
    CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    std::cout << "===== warm-up done =====" << std::endl << std::endl;
    
    tofile(outputs, "../Validation_py/c"); // ouputs data to files
    
    /*
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

    tofile(outputs, "../Validation_py/c"); // ouputs data to files
    */

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}


static ITensor* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
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
    return scale_1->getOutput(0);
}

static ITensor* addRepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname, bool rbr_identity)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* id_out;
    if (rbr_identity)
        id_out = addBatchNorm2d(network, weightMap, *input, lname + "rbr_identity", 1e-3);

    //rbr_dense
    IConvolutionLayer* conv1;
    if (rbr_identity) {
        conv1 = network->addConvolutionNd(*input, outch, DimsHW{ 3, 3 }, weightMap[lname + "rbr_dense.conv.weight"], emptywts);
        conv1->setStrideNd(DimsHW{ 1, 1 });
        conv1->setPaddingNd(DimsHW{ 1, 1 });
    }
    else {
        conv1 = network->addConvolutionNd(*input, outch, DimsHW{ 3, 3 }, weightMap[lname + "rbr_dense.conv.weight"], emptywts);
        conv1->setStrideNd(DimsHW{ 2, 2 });
        conv1->setPaddingNd(DimsHW{ 1, 1 });
    }
    ITensor* rbr_dense = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "rbr_dense.bn", 1e-3);

    //rbr_1x1
    IConvolutionLayer* conv2;
    if (rbr_identity) {
        conv2 = network->addConvolutionNd(*input, outch, DimsHW{ 1, 1 }, weightMap[lname + "rbr_1x1.conv.weight"], emptywts);
        conv2->setStrideNd(DimsHW{ 1, 1 });
    }
    else {
        conv2 = network->addConvolutionNd(*input, outch, DimsHW{ 1, 1 }, weightMap[lname + "rbr_1x1.conv.weight"], emptywts);
        conv2->setStrideNd(DimsHW{ 2, 2 });
    }
    ITensor* rbr_1x1 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "rbr_1x1.bn", 1e-3);

    IElementWiseLayer* elt_sum0 = network->addElementWise(*rbr_dense, *rbr_1x1, ElementWiseOperation::kSUM);

    if (rbr_identity)
        elt_sum0 = network->addElementWise(*elt_sum0->getOutput(0), *id_out, ElementWiseOperation::kSUM);

    auto* nonlinearity = network->addActivation(*elt_sum0->getOutput(0), ActivationType::kRELU);
    return nonlinearity->getOutput(0);
}