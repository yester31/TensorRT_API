//#include "cuda_runtime_api.h"
//#include "NvInfer.h"
//#include "NvInferPlugin.h"
//#include <string>
/*using namespace std;
using namespace nvinfer1;

int plugin_layer(INetworkDefinition* network, ITensor* data) {

	SPreproc preproc{
	1,2,3,4
	};

	IPluginCreator* creator = getPluginRegistry()->getPluginCreator("preproc", "4");
	IPluginV2 *plugin = creator->createPlugin("layerName(class)", (PluginFieldCollection*)&preproc);
	IPluginV2Layer* plugin_layer = network->addPluginV2(&data, 1, *plugin);
	plugin_layer->setName("layer(instance)");
	data = plugin_layer->getOutput(0);

}


int SPreprocPluginV2::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
	uint8_t* input = (uint8_t*)inputs[0];
	float* output = (float*)outputs[0];

	const int H = mPreproc.H;
	const int W = mPreproc.W;

	void preproc_hwc3_bgr8_zp1(float* output, unsigned char*input, int batchSize, int height, int width, cudaStream_t stream);
	preproc_hwc3_bgr8_zp1((float*)outputs[0], (unsigned char*)inputs[0], batchSize, H, W, stream);
}*/