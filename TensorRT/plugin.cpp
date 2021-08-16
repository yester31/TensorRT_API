#include "plugin.h"
REGISTER_TENSORRT_PLUGIN(SPreprocPluginV2Creator);

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


int SPreprocPluginV2::enqueue(int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	uint8_t* input = (uint8_t*)inputs[0];
	float* output = (float*)outputs[0];
}