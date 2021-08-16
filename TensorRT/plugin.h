#pragma once
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <string>
#include "NvInferRuntimeCommon.h"

using namespace std;
using namespace nvinfer1;

struct SPreproc {
	int N;
	int C;
	int H;
	int W;
	
};

class SPreprocPluginV2 : public IPluginV2IOExt
{
public:
	SPreprocPluginV2(const SPreproc& arg)
	{
		mPreproc = arg;
	}

	SPreprocPluginV2(const void* data, size_t length)
	{
		const char* d = static_cast<const char*>(data);
		const char* const a = d;
		mPreproc = read<SPreproc>(d);
		assert(d == a + length);
	}
	SPreprocPluginV2() = delete;

	virtual ~SPreprocPluginV2() {}

public:
	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		return Dims3(mPreproc.C, mPreproc.H, mPreproc.W);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}
	int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	size_t getSerializationSize() const override
	{
		size_t serializationSize = 0;
		serializationSize += sizeof(mPreproc);
		return serializationSize;
	}

	void serialize(void* buffer) const override
	{
		char* d = static_cast<char*>(buffer);
		const char* const a = d;
		write(d, mPreproc);
		assert(d == a + getSerializationSize());
	}

	void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
	{
	}
	//! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
	bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
	{
		assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
		bool condition = inOut[pos].format == TensorFormat::kLINEAR;
		condition &= inOut[pos].type != DataType::kINT32;
		condition &= inOut[pos].type == inOut[0].type;
		return condition;
	}
	DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override
	{
		assert(inputTypes && nbInputs == 1);
		//kFLOAT = 0, //!< FP32 format.
		//kHALF = 1,  //!< FP16 format.
		//kINT8 = 2,  //!< quantized INT8 format.
		//kINT32 = 3  //!< INT32 format.
		return DataType::kFLOAT;
	}

	const char* getPluginType() const override
	{
		return "preproc";
	}

	const char* getPluginVersion() const override
	{
		return "4";
	}

	void destroy() override
	{
		delete this;
	}

	IPluginV2Ext* clone() const override
	{
		auto* plugin = new SPreprocPluginV2(*this);
		return plugin;
	}

	void setPluginNamespace(const char* libNamespace) override
	{
		mNamespace = libNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mNamespace.data();
	}

	bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override
	{
		return false;
	}

	bool canBroadcastInputAcrossBatch(int inputIndex) const override
	{
		return false;
	}

private:
	template <typename T>
	void write(char*& buffer, const T& val) const
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template <typename T>
	T read(const char*& buffer) const
	{
		T val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
		return val;
	}

private:
	SPreproc mPreproc;
	string mNamespace;
};

class SPreprocPluginV2Creator : public IPluginCreator
{
public:
	const char* getPluginName() const override
	{
		return "preproc";
	}

	const char* getPluginVersion() const override
	{
		return "4";
	}

	const PluginFieldCollection* getFieldNames() override { return nullptr; } // 사용안함
	IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
	{
		SPreprocPluginV2* plugin = new SPreprocPluginV2(*(SPreproc*)fc);
		mPluginName = name;
		return plugin;
	}

	IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
	{
		auto plugin = new SPreprocPluginV2(serialData, serialLength);
		mPluginName = name;
		return plugin;
	}

	void setPluginNamespace(const char* libNamespace) override
	{
		mNamespace = libNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mNamespace.c_str();
	}

private:
	string mNamespace;
	string mPluginName;
};
