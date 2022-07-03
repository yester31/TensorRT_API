#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>
#include <vector>
#include <iostream>

using namespace std;

__global__ void kernel_yololayer_cu(
	float* output, 
	float* input, 
	float* anchor_grid, 
	int height, int width, int channel, int grid_stride, int out_size,
	int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	int c_idx = pos % channel;
	int idx = pos / channel;
	int o_idx = idx % out_size;
	int b_idx = idx / out_size;

	int w_idx = idx % width;
	idx /= width;
	int h_idx = idx % height;
	idx /= height;
	int ic_idx = (idx % 3) * 2;

	int g_idx = b_idx * out_size * channel + o_idx * channel;
	int g_idx2 = b_idx * out_size * 6 + o_idx * 6;

	output[g_idx2] = (input[g_idx] * 2 - 0.5 + w_idx) * grid_stride;
	output[g_idx2 + 1] = (input[g_idx + 1] * 2 - 0.5 + h_idx) * grid_stride;
	output[g_idx2 + 2] = input[g_idx + 2] * input[g_idx + 2] * 4 * anchor_grid[ic_idx] * grid_stride;
	output[g_idx2 + 3] = input[g_idx + 3] * input[g_idx + 3] * 4 * anchor_grid[ic_idx + 1] * grid_stride;

	float box_prob = input[g_idx + 4];
	if (box_prob < 0.1f) {
		output[g_idx2 + 4] = 0;
		output[g_idx2 + 5] = -1;
	}
	else {
		int class_id = 0;
		float max_cls_prob = 0.0;
		for (int i = 5; i < channel; ++i) {
			float p = input[g_idx + i];
			if (p > max_cls_prob) {
				max_cls_prob = p;
				class_id = i - 5;
			}
		}
		output[g_idx2 + 4] = box_prob * max_cls_prob;
		output[g_idx2 + 5] = class_id;
	}
	
}
void yololayer_cu(float* output, float* input, float* anchor_grid, int batchSize, int height, int width, int CLASS_NUM, int Grid_stride, cudaStream_t stream)
{
	int tcount = batchSize * height * width * 3 * (CLASS_NUM + 5);
	int blocks = 512;
	int grids = (tcount - 1) / blocks + 1;

	kernel_yololayer_cu << <grids, blocks, 0, stream >> > (output, input, anchor_grid, height, width, CLASS_NUM + 5, Grid_stride, height * width * 3, tcount);
}