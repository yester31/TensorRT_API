#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 536;
static const int INPUT_W = 640;
static const int INPUT_C = 3;

static const int CLASS_NUM = 80;
static const int ANCHORS = 1;
static const int MAX_OUTPUT_BBOX_COUNT = 100;

static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static ITensor* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
static ITensor* addRepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname, bool rbr_identity);
static ITensor* addSimSPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, std::string lname);
static ITensor* addRepVGGBlock2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname);
static ITensor* addSimConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname);
static ITensor* addSimConv2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname);
static ITensor* addSilu(INetworkDefinition *network, ITensor* input);
static ITensor *addReahspe(INetworkDefinition *network, std::vector<int> reshape_dims, ITensor *input);
static ITensor *addRegress(INetworkDefinition *network, ITensor *reg_output0, ITensor *obj_output0, ITensor *cls_output0, int stride);

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
    ITensor* X2 = addRepVGGBlock(network, weightMap, ERBlock_3_3, 128, "backbone.ERBlock_3.1.block.2.", true);
    // ERBlock_4
    ITensor* ERBlock_4_0 = addRepVGGBlock(network, weightMap, X2, 256, "backbone.ERBlock_4.0.", false);
    ITensor* ERBlock_4_1 = addRepVGGBlock(network, weightMap, ERBlock_4_0, 256, "backbone.ERBlock_4.1.conv1.", true);
    ITensor* ERBlock_4_2 = addRepVGGBlock(network, weightMap, ERBlock_4_1, 256, "backbone.ERBlock_4.1.block.0.", true);
    ITensor* ERBlock_4_3 = addRepVGGBlock(network, weightMap, ERBlock_4_2, 256, "backbone.ERBlock_4.1.block.1.", true);
    ITensor* ERBlock_4_4 = addRepVGGBlock(network, weightMap, ERBlock_4_3, 256, "backbone.ERBlock_4.1.block.2.", true);
    ITensor* ERBlock_4_5 = addRepVGGBlock(network, weightMap, ERBlock_4_4, 256, "backbone.ERBlock_4.1.block.3.", true);
    ITensor* X1 = addRepVGGBlock(network, weightMap, ERBlock_4_5, 256, "backbone.ERBlock_4.1.block.4.", true);
    // ERBlock_5
    ITensor* ERBlock_5_0 = addRepVGGBlock(network, weightMap, X1, 512, "backbone.ERBlock_5.0.", false);
    ITensor* ERBlock_5_1 = addRepVGGBlock(network, weightMap, ERBlock_5_0, 512, "backbone.ERBlock_5.1.conv1.", true);
    ITensor* ERBlock_5_2 = addRepVGGBlock(network, weightMap, ERBlock_5_1, 512, "backbone.ERBlock_5.1.block.0.", true);
    ITensor* X0 = addSimSPPF(network, weightMap, ERBlock_5_2, "backbone.ERBlock_5.2.");
    // backbone

    // neck
    ITensor* fpn_out0 = addSimConv(network, weightMap, X0, 128, "neck.reduce_layer0.");
    IDeconvolutionLayer* upsample_feat0 = network->addDeconvolutionNd(*fpn_out0, 128, DimsHW{ 2, 2 }, weightMap["neck.upsample0.upsample_transpose.weight"], weightMap["neck.upsample0.upsample_transpose.bias"]);  //nn.ConvTranspose2d
    upsample_feat0->setStrideNd(DimsHW{ 2, 2 });
    ITensor* inputs2_0[] = { upsample_feat0->getOutput(0), X1};
    auto f_concat_layer0 = network->addConcatenation(inputs2_0, 2);
    ITensor* f_out0 = addRepVGGBlock2(network, weightMap, f_concat_layer0->getOutput(0), 128, "neck.Rep_p4.conv1.");
    ITensor* f_out0_1 = addRepVGGBlock(network, weightMap, f_out0, 128, "neck.Rep_p4.block.0.", true);
    ITensor* f_out0_2 = addRepVGGBlock(network, weightMap, f_out0_1, 128, "neck.Rep_p4.block.1.", true);
    ITensor* f_out0_3 = addRepVGGBlock(network, weightMap, f_out0_2, 128, "neck.Rep_p4.block.2.", true);

    ITensor* fpn_out1 = addSimConv(network, weightMap, f_out0_3, 64, "neck.reduce_layer1.");
    IDeconvolutionLayer* upsample_feat1 = network->addDeconvolutionNd(*fpn_out1, 64, DimsHW{ 2, 2 }, weightMap["neck.upsample1.upsample_transpose.weight"], weightMap["neck.upsample1.upsample_transpose.bias"]);  //nn.ConvTranspose2d
    upsample_feat1->setStrideNd(DimsHW{ 2, 2 });
    ITensor* inputs2_1[] = { upsample_feat1->getOutput(0), X2 };
    auto f_concat_layer1 = network->addConcatenation(inputs2_1, 2);
    ITensor* pan_out2 = addRepVGGBlock2(network, weightMap, f_concat_layer1->getOutput(0), 64, "neck.Rep_p3.conv1.");
    ITensor* pan_out2_1 = addRepVGGBlock(network, weightMap, pan_out2, 64, "neck.Rep_p3.block.0.", true);
    ITensor* pan_out2_2 = addRepVGGBlock(network, weightMap, pan_out2_1, 64, "neck.Rep_p3.block.1.", true);
    ITensor* pan_out2_3 = addRepVGGBlock(network, weightMap, pan_out2_2, 64, "neck.Rep_p3.block.2.", true);
    
    ITensor* down_feat1 = addSimConv2(network, weightMap, pan_out2_3, 64, "neck.downsample2.");
    ITensor* inputs2_2[] = { down_feat1, fpn_out1 };
    auto p_concat_layer1 = network->addConcatenation(inputs2_2, 2);
    ITensor* pan_out1 = addRepVGGBlock(network, weightMap, p_concat_layer1->getOutput(0), 128, "neck.Rep_n3.conv1.", true);
    ITensor* pan_out1_1 = addRepVGGBlock(network, weightMap, pan_out1, 128, "neck.Rep_n3.block.0.", true);
    ITensor* pan_out1_2 = addRepVGGBlock(network, weightMap, pan_out1_1, 128, "neck.Rep_n3.block.1.", true);
    ITensor* pan_out1_3 = addRepVGGBlock(network, weightMap, pan_out1_2, 128, "neck.Rep_n3.block.2.", true);
    
    ITensor* down_feat0 = addSimConv2(network, weightMap, pan_out1_3, 128, "neck.downsample1.");
    ITensor* inputs2_3[] = { down_feat0, fpn_out0 };
    auto p_concat_layer2 = network->addConcatenation(inputs2_3, 2);
    ITensor* pan_out0 = addRepVGGBlock(network, weightMap, p_concat_layer2->getOutput(0), 256, "neck.Rep_n4.conv1.", true);
    ITensor* pan_out0_1 = addRepVGGBlock(network, weightMap, pan_out0, 256, "neck.Rep_n4.block.0.", true);
    ITensor* pan_out0_2 = addRepVGGBlock(network, weightMap, pan_out0_1, 256, "neck.Rep_n4.block.1.", true);
    ITensor* pan_out0_3 = addRepVGGBlock(network, weightMap, pan_out0_2, 256, "neck.Rep_n4.block.2.", true);
    // neck

    //outputs = [pan_out2_3, pan_out1_3, pan_out0_3]
    //show_dims(pan_out2_3); // 64 68 80
    //show_dims(pan_out1_3); // 128, 34, 40
    //show_dims(pan_out0_3); // 256, 17, 20
    // detect
    // 0
    IConvolutionLayer* stems0_conv = network->addConvolutionNd(*pan_out2_3, 64, DimsHW{ 1, 1 }, weightMap["detect.stems.0.conv.weight"], weightMap["detect.stems.0.conv.bias"]);
    stems0_conv->setStrideNd(DimsHW{ 1, 1 });
    ITensor* stems0 = addSilu(network, stems0_conv->getOutput(0));

    IConvolutionLayer* cls_convs0 = network->addConvolutionNd(*stems0, 64, DimsHW{ 3, 3 }, weightMap["detect.cls_convs.0.conv.weight"], weightMap["detect.cls_convs.0.conv.bias"]);
    cls_convs0->setStrideNd(DimsHW{ 1, 1 });
    cls_convs0->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* cls_feat0 = addSilu(network, cls_convs0->getOutput(0));

    IConvolutionLayer* cls_preds0 = network->addConvolutionNd(*cls_feat0, CLASS_NUM, DimsHW{ 1, 1 }, weightMap["detect.cls_preds.0.weight"], weightMap["detect.cls_preds.0.bias"]);
    cls_preds0->setStrideNd(DimsHW{ 1, 1 });
    ITensor* cls_output0 = cls_preds0->getOutput(0);
    cls_output0 = network->addActivation(*cls_output0, ActivationType::kSIGMOID)->getOutput(0);

    IConvolutionLayer* reg_convs0 = network->addConvolutionNd(*stems0, 64, DimsHW{ 3, 3 }, weightMap["detect.reg_convs.0.conv.weight"], weightMap["detect.reg_convs.0.conv.bias"]);
    reg_convs0->setStrideNd(DimsHW{ 1, 1 });
    reg_convs0->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* reg_feat0 = addSilu(network, reg_convs0->getOutput(0));

    IConvolutionLayer* reg_preds0 = network->addConvolutionNd(*reg_feat0, 4, DimsHW{ 1, 1 }, weightMap["detect.reg_preds.0.weight"], weightMap["detect.reg_preds.0.bias"]);
    reg_preds0->setStrideNd(DimsHW{ 1, 1 });
    ITensor* reg_output0 = reg_preds0->getOutput(0);

    IConvolutionLayer* obj_preds0 = network->addConvolutionNd(*reg_feat0, 1, DimsHW{ 1, 1 }, weightMap["detect.obj_preds.0.weight"], weightMap["detect.obj_preds.0.bias"]);
    obj_preds0->setStrideNd(DimsHW{ 1, 1 });
    ITensor* obj_output0 = obj_preds0->getOutput(0);
    obj_output0 = network->addActivation(*obj_output0, ActivationType::kSIGMOID)->getOutput(0);

    ITensor * ny0 = addRegress(network, reg_output0, obj_output0, cls_output0, 8);
    
    // 1
    IConvolutionLayer* stems1_conv = network->addConvolutionNd(*pan_out1_3, 128, DimsHW{ 1, 1 }, weightMap["detect.stems.1.conv.weight"], weightMap["detect.stems.1.conv.bias"]);
    stems1_conv->setStrideNd(DimsHW{ 1, 1 });
    ITensor* stems1 = addSilu(network, stems1_conv->getOutput(0));

    IConvolutionLayer* cls_convs1 = network->addConvolutionNd(*stems1, 128, DimsHW{ 3, 3 }, weightMap["detect.cls_convs.1.conv.weight"], weightMap["detect.cls_convs.1.conv.bias"]);
    cls_convs1->setStrideNd(DimsHW{ 1, 1 });
    cls_convs1->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* cls_feat1 = addSilu(network, cls_convs1->getOutput(0));

    IConvolutionLayer* cls_preds1 = network->addConvolutionNd(*cls_feat1, CLASS_NUM, DimsHW{ 1, 1 }, weightMap["detect.cls_preds.1.weight"], weightMap["detect.cls_preds.1.bias"]);
    cls_preds1->setStrideNd(DimsHW{ 1, 1 });
    ITensor* cls_output1 = cls_preds1->getOutput(0);
    cls_output1 = network->addActivation(*cls_output1, ActivationType::kSIGMOID)->getOutput(0);

    IConvolutionLayer* reg_convs1 = network->addConvolutionNd(*stems1, 128, DimsHW{ 3, 3 }, weightMap["detect.reg_convs.1.conv.weight"], weightMap["detect.reg_convs.1.conv.bias"]);
    reg_convs1->setStrideNd(DimsHW{ 1, 1 });
    reg_convs1->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* reg_feat1 = addSilu(network, reg_convs1->getOutput(0));

    IConvolutionLayer* reg_preds1 = network->addConvolutionNd(*reg_feat1, 4, DimsHW{ 1, 1 }, weightMap["detect.reg_preds.1.weight"], weightMap["detect.reg_preds.1.bias"]);
    reg_preds1->setStrideNd(DimsHW{ 1, 1 });
    ITensor* reg_output1 = reg_preds1->getOutput(0);

    IConvolutionLayer* obj_preds1 = network->addConvolutionNd(*reg_feat1, 1, DimsHW{ 1, 1 }, weightMap["detect.obj_preds.1.weight"], weightMap["detect.obj_preds.1.bias"]);
    obj_preds1->setStrideNd(DimsHW{ 1, 1 });
    ITensor* obj_output1 = obj_preds1->getOutput(0);
    obj_output1 = network->addActivation(*obj_output1, ActivationType::kSIGMOID)->getOutput(0);

    ITensor * ny1 = addRegress(network, reg_output1, obj_output1, cls_output1, 16);

    //// 2
    IConvolutionLayer* stems2_conv = network->addConvolutionNd(*pan_out0_3, 256, DimsHW{ 1, 1 }, weightMap["detect.stems.2.conv.weight"], weightMap["detect.stems.2.conv.bias"]);
    stems2_conv->setStrideNd(DimsHW{ 1, 1 });
    ITensor* stems2 = addSilu(network, stems2_conv->getOutput(0));

    IConvolutionLayer* cls_convs2 = network->addConvolutionNd(*stems2, 256, DimsHW{ 3, 3 }, weightMap["detect.cls_convs.2.conv.weight"], weightMap["detect.cls_convs.2.conv.bias"]);
    cls_convs2->setStrideNd(DimsHW{ 1, 1 });
    cls_convs2->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* cls_feat2 = addSilu(network, cls_convs2->getOutput(0));

    IConvolutionLayer* cls_preds2 = network->addConvolutionNd(*cls_feat2, CLASS_NUM, DimsHW{ 1, 1 }, weightMap["detect.cls_preds.2.weight"], weightMap["detect.cls_preds.2.bias"]);
    cls_preds2->setStrideNd(DimsHW{ 1, 1 });
    ITensor* cls_output2 = cls_preds2->getOutput(0);
    cls_output2 = network->addActivation(*cls_output2, ActivationType::kSIGMOID)->getOutput(0);

    IConvolutionLayer* reg_convs2 = network->addConvolutionNd(*stems2, 256, DimsHW{ 3, 3 }, weightMap["detect.reg_convs.2.conv.weight"], weightMap["detect.reg_convs.2.conv.bias"]);
    reg_convs2->setStrideNd(DimsHW{ 1, 1 });
    reg_convs2->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* reg_feat2 = addSilu(network, reg_convs2->getOutput(0));

    IConvolutionLayer* reg_preds2 = network->addConvolutionNd(*reg_feat2, 4, DimsHW{ 1, 1 }, weightMap["detect.reg_preds.2.weight"], weightMap["detect.reg_preds.2.bias"]);
    reg_preds2->setStrideNd(DimsHW{ 1, 1 });
    ITensor* reg_output2 = reg_preds2->getOutput(0);

    IConvolutionLayer* obj_preds2 = network->addConvolutionNd(*reg_feat2, 1, DimsHW{ 1, 1 }, weightMap["detect.obj_preds.2.weight"], weightMap["detect.obj_preds.2.bias"]);
    obj_preds2->setStrideNd(DimsHW{ 1, 1 });
    ITensor* obj_output2 = obj_preds2->getOutput(0);
    obj_output2 = network->addActivation(*obj_output2, ActivationType::kSIGMOID)->getOutput(0);

    ITensor * ny2 = addRegress(network, reg_output2, obj_output2, cls_output2, 32);

    ITensor* z_input[] = { ny0, ny1, ny2 };
    IConcatenationLayer* zconcat = network->addConcatenation(z_input, 3);
    zconcat->setAxis(0);
    ITensor* z = zconcat->getOutput(0);

    auto slice_layer = network->addSlice(*z, Dims2(0, 4), Dims2(z->getDimensions().d[0], 1), Dims2(1, 1));
    auto sort_layer = network->addTopK(*slice_layer->getOutput(0), TopKOperation::kMAX, MAX_OUTPUT_BBOX_COUNT, 1 << 0);
    auto shuffle_layer = network->addShuffle(*sort_layer->getOutput(1));
    Dims dims_shape; dims_shape.nbDims = 1; dims_shape.d[0] = MAX_OUTPUT_BBOX_COUNT;
    shuffle_layer->setReshapeDimensions(dims_shape);
    auto gather_layer = network->addGather(*z, *shuffle_layer->getOutput(0), 0);
    ITensor* final_tensor = gather_layer->getOutput(0);

    // detect

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
        std::cout << "==== precision int8 ====" << std::endl << std::endl;
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, 0, "../data_calib/", "../Int8_calib_table/yolov6s_int8_calib.table", INPUT_BLOB_NAME);
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
    unsigned int maxBatchSize = 1;          // batch size 
    bool serialize = false;                 // TensorRT Model Serialize flag(true : generate engine, false : if no engine file, generate engine )
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
    int OUTPUT_SIZE = 1 * MAX_OUTPUT_BBOX_COUNT * 85;
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
    uint64_t iter_count = 1000;

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
    std::cout << "Avg duration time with data transfer : " << (float)dur_time / iter_count << " [milliseconds]" << std::endl;
    std::cout << "FPS : " << 1000.f / ((float)dur_time / iter_count) << " [frame/sec]" << std::endl;
    std::cout << "===== TensorRT Model Calculate done =====" << std::endl;
    std::cout << "==================================================" << std::endl;

    tofile(outputs, "../Validation_py/c"); // ouputs data to files

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

static ITensor* addRepVGGBlock2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    //rbr_dense
    IConvolutionLayer* conv1 = network->addConvolutionNd(*input, outch, DimsHW{ 3, 3 }, weightMap[lname + "rbr_dense.conv.weight"], emptywts);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* rbr_dense = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "rbr_dense.bn", 1e-3);

    //rbr_1x1
    IConvolutionLayer* conv2 = network->addConvolutionNd(*input, outch, DimsHW{ 1, 1 }, weightMap[lname + "rbr_1x1.conv.weight"], emptywts);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    ITensor* rbr_1x1 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "rbr_1x1.bn", 1e-3);

    IElementWiseLayer* elt_sum0 = network->addElementWise(*rbr_dense, *rbr_1x1, ElementWiseOperation::kSUM);

    auto* nonlinearity = network->addActivation(*elt_sum0->getOutput(0), ActivationType::kRELU);
    return nonlinearity->getOutput(0);
}


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

static ITensor* addSimConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(*input, outch, DimsHW{ 1, 1 }, weightMap[lname + "conv.weight"], emptywts);
    conv1->setStrideNd(DimsHW{ 1, 1 });

    ITensor* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-3);
    auto* nonlinearity0 = network->addActivation(*bn1, ActivationType::kRELU);
    return nonlinearity0->getOutput(0);
}

static ITensor* addSimConv2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, int outch, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(*input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv.weight"], emptywts);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });

    ITensor* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-3);
    auto* nonlinearity0 = network->addActivation(*bn1, ActivationType::kRELU);
    return nonlinearity0->getOutput(0);
}

static ITensor* addSilu(INetworkDefinition *network, ITensor* input)
{
    // silu = x * sigmoid
    auto sig = network->addActivation(*input, ActivationType::kSIGMOID);
    auto ew = network->addElementWise(*input, *sig->getOutput(0), ElementWiseOperation::kPROD);
    return ew->getOutput(0);
}


static ITensor *addReahspe(INetworkDefinition *network, std::vector<int> reshape_dims, ITensor *input)
{
    IShuffleLayer *shuffle_layer = network->addShuffle(*input);
    Dims shape_dims;
    shape_dims.nbDims = (int)reshape_dims.size();
    memcpy(shape_dims.d, reshape_dims.data(), reshape_dims.size() * sizeof(int));
    shuffle_layer->setReshapeDimensions(shape_dims);
    return shuffle_layer->getOutput(0);
}

static ITensor *addRegress(INetworkDefinition *network, ITensor *reg_output0, ITensor *obj_output0, ITensor *cls_output0, int stride)
{
    ITensor* candi0[] = { reg_output0, obj_output0, cls_output0 };
    ITensor* concat0 = network->addConcatenation(candi0, 3)->getOutput(0);
    IShuffleLayer* shuffle0 = network->addShuffle(*concat0);
    shuffle0->setReshapeDimensions(Dims4(ANCHORS, 4 + 1 + CLASS_NUM, concat0->getDimensions().d[1], concat0->getDimensions().d[2]));
    std::vector<int> trans_dims{ 0, 2, 3, 1 };
    Permutation s_trans_dims; memcpy(s_trans_dims.order, trans_dims.data(), trans_dims.size() * sizeof(int));
    shuffle0->setSecondTranspose(s_trans_dims);
    ITensor* y0 = shuffle0->getOutput(0);

    std::vector<float> grids_vec0;
    //torch.meshgrid
    for (int y_idx = 0; y_idx < concat0->getDimensions().d[1]; y_idx++) {
        for (int x_idx = 0; x_idx < concat0->getDimensions().d[2]; x_idx++) {
            grids_vec0.push_back(x_idx);
            grids_vec0.push_back(y_idx);
        }
    }

    float *grids_ptr0 = reinterpret_cast<float*>(malloc(sizeof(float) * grids_vec0.size()));
    memcpy(grids_ptr0, grids_vec0.data(), grids_vec0.size() * sizeof(float));
    Weights grids_wt0{ DataType::kFLOAT, grids_ptr0, grids_vec0.size() };
    ITensor* grids_tensor0 = network->addConstant(Dims4{ ANCHORS, concat0->getDimensions().d[1], concat0->getDimensions().d[2], 2 }, grids_wt0)->getOutput(0);

    float *stride_ptr0 = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    *stride_ptr0 = stride;
    Weights stride_wt0{ DataType::kFLOAT, stride_ptr0, 1 };
    nvinfer1::Dims strid_dim;
    strid_dim.nbDims = 1;
    strid_dim.d[0] = 1;
    ITensor* strid0 = network->addConstant(strid_dim, stride_wt0)->getOutput(0);
    strid0 = addReahspe(network, { 1,1,1,1 }, strid0);

    ITensor* box01_0 = network->addSlice(*y0, Dims4(0, 0, 0, 0), Dims4(ANCHORS, concat0->getDimensions().d[1], concat0->getDimensions().d[2], 2), Dims4{ 1, 1, 1, 1 })->getOutput(0);
    ITensor* box01_sum0 = network->addElementWise(*box01_0, *grids_tensor0, ElementWiseOperation::kSUM)->getOutput(0);
    ITensor* box23_0 = network->addSlice(*y0, Dims4(0, 0, 0, 2), Dims4(ANCHORS, concat0->getDimensions().d[1], concat0->getDimensions().d[2], 2), Dims4{ 1, 1, 1, 1 })->getOutput(0);
    ITensor* box23_exp0 = network->addUnary(*box23_0, UnaryOperation::kEXP)->getOutput(0);
    ITensor* box_input0[] = { box01_sum0, box23_exp0 };
    IConcatenationLayer* boxconcat0 = network->addConcatenation(box_input0, 2);
    boxconcat0->setAxis(3);
    ITensor* box0 = boxconcat0->getOutput(0);
    ITensor* nbox0 = network->addElementWise(*box0, *strid0, ElementWiseOperation::kPROD)->getOutput(0);

    ITensor* objcls0 = network->addSlice(*y0, Dims4(0, 0, 0, 4), Dims4(ANCHORS, concat0->getDimensions().d[1], concat0->getDimensions().d[2], 1 + CLASS_NUM), Dims4{ 1, 1, 1, 1 })->getOutput(0);
    ITensor* y0_input0[] = { nbox0, objcls0 };
    IConcatenationLayer* nyconcat0 = network->addConcatenation(y0_input0, 2);
    nyconcat0->setAxis(3);
    ITensor* ny0 = addReahspe(network, { -1, 4 + 1 + CLASS_NUM }, nyconcat0->getOutput(0));
    return ny0;
}

//static ITensor *addRegress2(INetworkDefinition *network, ITensor *reg_output0, ITensor *obj_output0, ITensor *cls_output0, int stride, int minimum = 100)
//{
//    IShuffleLayer* shuffle0 = network->addShuffle(*cls_output0);
//    std::vector<int> trans_dims{ 1, 2, 0 };
//    Permutation f_trans_dims; memcpy(f_trans_dims.order, trans_dims.data(), trans_dims.size() * sizeof(int));
//    shuffle0->setFirstTranspose(f_trans_dims);
//    shuffle0->setReshapeDimensions(Dims2(cls_output0->getDimensions().d[1] * cls_output0->getDimensions().d[2], CLASS_NUM));
//    ITensor* cls_probs = shuffle0->getOutput(0); // 80, 68, 80 -> (68, 80, 80) -> (68 * 80, 80)
//    IShuffleLayer* shuffle1 = network->addShuffle(*reg_output0);
//    shuffle1->setFirstTranspose(f_trans_dims);
//    shuffle1->setReshapeDimensions(Dims2(reg_output0->getDimensions().d[1] * reg_output0->getDimensions().d[2], 4));
//    ITensor* rois = shuffle1->getOutput(0); // 4, 68, 80 -> (68, 80, 4) -> (68 * 80, 4)
//    ITensor* obj_ness = addReahspe(network, { cls_output0->getDimensions().d[1] * cls_output0->getDimensions().d[2], 1 }, obj_output0); // 1, 68, 80 -> (68 * 80, 1)
//
//    int minimum_pick = std::min(cls_probs->getDimensions().d[0], minimum);
//
//    ITopKLayer* top0 = network->addTopK(*obj_ness, TopKOperation::kMAX, minimum_pick, 1 << 0); // 1 << 0 -> axis = 0
//    ITensor* obj_conf = top0->getOutput(0); // [minimum_pick,1] sorted prob (*)
//    ITensor* index = top0->getOutput(1);    // [minimum_pick,1]
//
//    IShuffleLayer* shuffle1 = network->addShuffle(*index);
//    Dims shape_dims; shape_dims.nbDims = 1; shape_dims.d[0] = minimum_pick;
//    shuffle1->setReshapeDimensions(shape_dims);
//    index = shuffle1->getOutput(0);             // [minimum_pick,1] -> [minimum_pick]
//
//    IGatherLayer* gather0 = network->addGather(*cls_probs, *index, 0);
//    ITensor* cls_probs_sorted = gather0->getOutput(0); // [minimum_pick, 80] sorted by obj	
//
//    ITopKLayer* top1 = network->addTopK(*cls_probs_sorted, TopKOperation::kMAX, 1, 1 << 1); // 1 << 1 -> axis = 1
//    ITensor* cls_conf = top1->getOutput(0);     // [minimum_pick, 80] -> [minimum_pick, 1]
//    ITensor* cls_index = top1->getOutput(1);    // [minimum_pick, 1] class index (*)
//
//    IGatherLayer* gather1 = network->addGather(*rois, *index, 0);
//    ITensor* roi = gather1->getOutput(0);       // [minimum_pick, 4] bbox sorted by obj (*)
//
//    // conf = obj_conf * cls_conf
//
//    float *stride_ptr0 = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
//    *stride_ptr0 = stride;
//    Weights stride_wt0{ DataType::kFLOAT, stride_ptr0, 1 };
//    nvinfer1::Dims strid_dim;
//    strid_dim.nbDims = 1;
//    strid_dim.d[0] = 1;
//    ITensor* strid0 = network->addConstant(strid_dim, stride_wt0)->getOutput(0);
//    strid0 = addReahspe(network, { 1,1,1,1 }, strid0);
//
//    Yololayer2 yololayer2{ minimum_pick };
//    IPluginCreator* creator0 = getPluginRegistry()->getPluginCreator("yololayer2", "1");
//    IPluginV2 *plugin0 = creator0->createPlugin("yololayer2_plugin", (PluginFieldCollection*)&yololayer2);
//    std::vector<ITensor*> datas{ roi, obj_conf, cls_conf, cls_index, strid0 };
//    IPluginV2Layer* plugin_layer0 = network->addPluginV2(datas.data(), (int)datas.size(), *plugin0);
//
//    return plugin_layer0->getOutput(0);
//}