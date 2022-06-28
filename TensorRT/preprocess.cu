#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>
#include <vector>
#include <iostream>

using namespace std;

// 전처리 함수 0 (NHWC->NCHW, BGR->RGB, [0, 255]->[0.0, 1.0](Normalize))
__global__ void kernel_preprocess_0(
    float* output,				// [N,RGB,H,W]
    const unsigned char* input, // [N,H,W,BGR]
    const int batchSize, const int height, const int width, const int channel,
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

    int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

    output[pos] = input[g_idx] / 255.f;
}

void preprocess_cu_0(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
    int tcount = batchSize * height * width * channel;
    int block = 512;
    int grid = (tcount - 1) / block + 1;

    kernel_preprocess_0 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
}

// 전처리 함수 1 (NHWC->NCHW, BGR->RGB, [0, 255]->[0.0, 1.0], 
// Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
__constant__ float constMem_mean_std[6];
__global__ void kernel_preprocess_1(
    float* output,				// [N,RGB,H,W]
    const unsigned char* input, // [N,H,W,BGR]
    const int batchSize, const int height, const int width, const int channel,
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

    int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

    output[pos] = (input[g_idx] / 255.f - constMem_mean_std[c_idx]) / constMem_mean_std[c_idx + 3];
}

void preprocess_cu_1(float* output, unsigned char*input, int batchSize, int height, int width, int channel, std::vector<float> &mean_std, cudaStream_t stream)
{
    int tcount = batchSize * height * width * channel;
    int block = 512;
    int grid = (tcount - 1) / block + 1;

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    cudaMemcpyToSymbol(constMem_mean_std, mean_std.data(), sizeof(float) * 6);
    kernel_preprocess_1 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float time;
    //cudaEventElapsedTime(&time, start, stop);
    //std::cout << "elapsed time :: " << time << std::endl;
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    //elapsed time :: 0.635904 
    //elapsed time :: 0.599040 (cuda constant mem w data transfer)
    //elapsed time :: 0.492544 (cuda constant mem wo data transfer)

}

__device__ __forceinline__ float area_pixel_compute_source_index(float scale, int dst_index)
{
    float src_idx = scale * (dst_index + 0.5f) - 0.5f;
    return src_idx < 0 ? 0 : src_idx;
}
// 전처리 함수 3 (NHWC->NCHW, BGR->RGB, [0, 255]->[0.0, 1.0](Normalize), letterbox padding)
__global__ void kernel_preprocess_3(
    float* output,              // [N,RGB,P,Q]
    const unsigned char* input, // [N,H,W,BGR]
    float rheight, float rwidth,
    int P, int Q,   //OUTPUT(padded) for % 32
    int P0, int Q0, //RESIZE(resized)
    int H, int W,   //INPUT
    int pt, int pb, int pl, int pr, const int tcount)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= tcount) return;

    const int q_idx = pos % Q;
    int idx = pos / Q;
    const int p_idx = idx % P;
    idx /= P;
    const int c_idx = idx % 3;
    const int b_idx = idx / 3;

    if (q_idx < Q0 && p_idx < P0)
    {
        if (H == P0 && W == Q0) {
            int s_idx = b_idx * H * W * 3 + p_idx * W * 3 + q_idx * 3 + 2 - c_idx;
            output[pos + pt * Q + pl] = input[s_idx] / 255.f;
        }
        else {
            const float h1r = area_pixel_compute_source_index(rheight, p_idx);
            const int h1 = h1r;
            const int h1p = (h1 < H - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = 1.f - h1lambda;

            const float w1r = area_pixel_compute_source_index(rwidth, q_idx);
            const int w1 = w1r;
            const int w1p = (w1 < W - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = 1.f - w1lambda;

            int base = b_idx * H * W * 3 + h1 * W * 3 + w1 * 3 + 2 - c_idx;

            float val = h0lambda * (w0lambda * input[base] + w1lambda * input[base + w1p * 3]) +
                h1lambda * (w0lambda * input[base + h1p * W * 3] + w1lambda * input[base + h1p * W * 3 + w1p * 3]);
            output[pos + pt * Q + pl] = val / 255.f;
        }
    }
    else {
        if (pl <= q_idx && q_idx < Q0 + pl && pt <= p_idx && p_idx < P0 + pt) 
            output[b_idx * P * Q * 3 + c_idx * P * Q + ((P0 - 1 + pt) - p_idx) * Q + ((Q0 - 1 + pl) - q_idx)] = 114.f / 255.f;
        else 
            output[pos] = 114.f / 255.f;
    }
}

void preprocess_cu_3(float* output, unsigned char* input, int batchSize, int P, int Q, int P0, int Q0, int H, int W, int pt, int pb, int pl, int pr, cudaStream_t stream)
{
    int tcount = batchSize * P * Q * 3;
    int block = 512;
    int grid = (tcount - 1) / block + 1;

    float rheight = float(H) / P0;
    float rwidth = float(W) / Q0;

    kernel_preprocess_3 << <grid, block, 0, stream >> > (output, input, rheight, rwidth, P, Q, P0, Q0, H, W, pt, pb, pl, pr, tcount);
}