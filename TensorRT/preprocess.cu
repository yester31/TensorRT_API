#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>

using namespace std;

// 전처리 함수 0 (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
__global__ void kernel_preprocess_0(
	float* output,				// [N,RGB,H,W]
	const unsigned char* input, // [N,H,W,BGR]
	int batchSize, int height, int width, int channel,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int w_idx = pos % width;
	int idx = pos / width;
	const int h_idx = idx % height;
	idx /= height;
	const int c_idx = idx % channel;
	const int b_idx = idx / channel;

	int s_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

	output[pos] = input[s_idx] / 255.f;
	//output[pos] = input[s_idx];
}

void preprocess_cu_0(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
	int tcount = batchSize * height * width * channel;
	int block = 512;
	int grid = (tcount - 1) / block + 1;

	kernel_preprocess_0 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
}

// 전처리 함수 1 (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
__global__ void kernel_preprocess_1(
	float* output,				// [N,RGB,H,W]
	const unsigned char* input, // [N,H,W,BGR]
	int batchSize, int height, int width, int channel,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int w_idx = pos % width;
	int idx = pos / width;
	const int h_idx = idx % height;
	idx /= height;
	const int c_idx = idx % channel;
	const int b_idx = idx / channel;

	int s_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

	output[pos] = input[s_idx] / 255.f;
	//output[pos] = input[s_idx];
}

void preprocess_cu_1(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
	int tcount = batchSize * height * width * channel;
	int block = 512;
	int grid = (tcount - 1) / block + 1;

	kernel_preprocess_1 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
}