#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>
#include <vector>
#include <iostream>

using namespace std;

// 후처리 함수 0 (NCHW->NHWC, RGB->BGR, *255, ROUND, uint8)
__global__ void kernel_postprocess_0(
	uint8_t* output,// [N,H,W,RGB]
	float* input,	// [N,BGR,H,W]
	const int batchSize, const int height, const int width, const int channel,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int c_idx = pos % channel;
	int idx = pos / channel;
	const int w_idx = idx % width;
	idx /= width;
	const int h_idx = idx % height;
	const int b_idx = idx / height;

	int g_idx = b_idx * height * width * channel + (2 - c_idx)* height * width + h_idx * width + w_idx;
	float tt = input[g_idx] * 255.f;
	if (tt > 255)
		tt = 255;
	output[pos] = tt;
}

void postprocess_cu_0(unsigned char* output, float*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
	int tcount = batchSize * height * width * channel;
	int block = 512;
	int grid = (tcount - 1) / block + 1;

	kernel_postprocess_0 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
}


