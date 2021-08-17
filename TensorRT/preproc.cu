#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>

using namespace std;

__global__ void kernel_preproc_hwc3_bgr8_zp1(
	float* output, // [N,RGB,H,W]
	const unsigned char* input, // [N,H,W,BGR]
	int batchSize, int height, int width,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int w_idx = pos % width;
	int idx = pos / width;
	const int h_idx = idx % height;
	idx /= height;
	const int c_idx = idx % 3;
	const int b_idx = idx / 3;

	int s_idx = b_idx * height * width * 3 + h_idx * width * 3 + w_idx * 3 + 2 - c_idx;

	//output[pos] = input[s_idx] / 255.f;
	output[pos] = input[s_idx];
}


void preproc_hwc3_bgr8_zp1(float* output, unsigned char*input, int batchSize, int height, int width, cudaStream_t stream)
{
	int tcount = batchSize * height * width * 3;
	int block = 256;
	int grid = (tcount - 1) / block + 1;

	kernel_preproc_hwc3_bgr8_zp1 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, tcount);
}

const int WIDTH = 1024;							// total width is 1024* 1024
const int TILE_WIDTH = 32;						// block will be (TILE_WIDTH, TILE_WIDTH)
const int GRID_WIDTH = (WIDTH / TILE_WIDTH);	// grid will be (GRID_WIDTH, GRID_WIDTH)


// CUDA shared mem
__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
	//c[y][x] = sum_ka[y][k] * b[k][x]
	//c[y * WIDTH + x] = sum_ka[y * WIDTH + k] * b[k * WIDTH + x]
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int by = blockIdx.y; int bx = blockIdx.x;
	int ty = threadIdx.y; int tx = threadIdx.x;
	int gy = by * TILE_WIDTH + ty; // global y index
	int gx = bx * TILE_WIDTH + tx; // global x index
	float sum = 0.0F;
	for (register int m = 0; m < width / TILE_WIDTH; ++m) {
		//read into the shared memory blocks
		s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
		s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty)* width + gx];
		__syncthreads();
		//use the shared memory blocks to get the partial sum
		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[gy * width + gx] = sum;

}