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

//// 전처리 함수 1 (resize & letterbox, NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
//__global__ void kernel_preprocess_1(
//	float* output,				// [N,RGB,H,W]
//	const unsigned char* input, // [N,H,W,BGR]
//	int batchSize, int height, int width, int channel,
//	const int tcount)
//{
//	int pos = threadIdx.x + blockIdx.x * blockDim.x;
//	if (pos >= tcount) return;
//
//	const int w_idx = pos % width;
//	int idx = pos / width;
//	const int h_idx = idx % height;
//	idx /= height;
//	const int c_idx = idx % channel;
//	const int b_idx = idx / channel;
//
//	int s_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;
//
//	output[pos] = input[s_idx] / 255.f;
//	//output[pos] = input[s_idx];
//}
//
//float compute_scale(int input_size, int output_size, int align_corner)
//{
//	return align_corner ?
//		(input_size - 1.f) / (output_size - 1.f)
//		: float(input_size) / output_size;
//}
//
//void preprocess_cu_1(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
//{
//	int tcount = batchSize * height * width * channel;
//	int block = 512;
//	int grid = (tcount - 1) / block + 1;
//
//	//float hscale = compute_scale(H, P0, align_corner);
//	//float wscale = compute_scale(W, Q0, align_corner);
//
//	kernel_preprocess_1 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
//}
//// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu
//// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/UpSample.cuh
//__device__ __forceinline__ float compute_source_index(float scale, int dst_index, bool align_corner)
//{
//	if (align_corner) {
//		return scale * dst_index;
//	}
//	else {
//		float src_idx = scale * (dst_index + 0.5f) - 0.5f;
//		return src_idx < 0 ? 0 : src_idx;
//	}
//}
//__global__ void kernel_preproc_hwc3_bgr8_resize_zp1_3(
//	float* output,// [N,RGB,P,Q]
//	const unsigned char* input, // [N,H,W,BGR]
//	float hscale, float wscale,
//	int P, int Q,   //OUTPUT(padded)
//	int P0, int Q0, //RESIZE(resized)
//	int H, int W,	//INPUT
//	int align_corner,
//	int pl, int pr, int pt, int pb,
//	const int tcount)
//{
//	int pos = threadIdx.x + blockIdx.x * blockDim.x;
//	if (pos >= tcount) return;
//
//	const int q_idx = pos % Q;
//	int idx = pos / Q;
//	const int p_idx = idx % P;
//	idx /= P;
//	const int c_idx = idx % 3;
//	const int b_idx = idx / 3;
//
//	if (q_idx < Q0 && p_idx < P0)
//	{
//		if (H == P0 && W == Q0) { //no resize
//			int s_idx = b_idx * H * W * 3 + p_idx * W * 3 + q_idx * 3 + 2 - c_idx;
//			int pos2 = pos + pt * Q + pl;
//			output[pos2] = input[s_idx] / 255.f;
//		}
//		else {
//			const float h1r = compute_source_index(hscale, p_idx, align_corner);
//			const int h1 = h1r;
//			const int h1p = (h1 < H - 1) ? 1 : 0;
//			const float h1lambda = h1r - h1;
//			const float h0lambda = 1.f - h1lambda;
//
//			const float w1r = compute_source_index(wscale, q_idx, align_corner);
//			const int w1 = w1r;
//			const int w1p = (w1 < W - 1) ? 1 : 0;
//			const float w1lambda = w1r - w1;
//			const float w0lambda = 1.f - w1lambda;
//
//			int s_base = b_idx * H * W * 3 + h1 * W * 3 + w1 * 3 + 2 - c_idx;
//
//			float val =
//				h0lambda * (w0lambda * input[s_base] + w1lambda * input[s_base + w1p * 3]) +
//				h1lambda * (w0lambda * input[s_base + h1p * W * 3] + w1lambda * input[s_base + h1p * W * 3 + w1p * 3]);
//
//			int o_idx = pos + pt * Q + pl;
//			output[o_idx] = val / 255.f;
//		}
//	}
//	else {
//		if (pl <= q_idx && q_idx < Q0 + pl && pt <= p_idx && p_idx < P0 + pt) {
//			int q_idx_n = (Q0 - 1 + pl) - q_idx;
//			int p_idx_n = (P0 - 1 + pt) - p_idx;
//			int o_idx2 = b_idx * P * Q * 3 + c_idx * P * Q + p_idx_n * Q + q_idx_n;
//			output[o_idx2] = 114.f / 255.f;
//		}
//		else {
//			output[pos] = 114.f / 255.f;
//		}
//	}
//}