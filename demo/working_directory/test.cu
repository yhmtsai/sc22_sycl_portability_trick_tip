#include <iostream>
#include "trick/cooperative_groups.cuh"
#include "trick/dim3_t.hpp"

#define GET_QUEUE 0

template <typename ValueType, int subwarp_size = 16>
__global__ void reduction_kernel(const int num, ValueType* val)
{
    auto thread_block = this_thread_block_t();
    auto subwarp = tiled_partition_t<subwarp_size>(thread_block);
    // block_size only be 64
    __shared__ ValueType tmp[64];
    auto tid = threadIdx.x;
    auto local_data = val[tid];
#pragma unroll
    for (int bitmask = 1; bitmask < subwarp.size(); bitmask <<= 1) {
        const auto remote_data = subwarp.shfl_xor(local_data, bitmask);
        local_data = local_data + remote_data;
    }
    tmp[tid] = local_data;
    __syncthreads();
    if (tid < 32) {
        tmp[tid] += tmp[tid + 32];
    }
    __syncthreads();
    val[tid] = tmp[tid];
}


template <typename ValueType, int subwarp_size = 16>
void reduction_kernel_call(dim3_t grid, dim3_t block, unsigned int dynamic_shared_memory,
                      cudaStream_t queue, const int num, ValueType* val)
{
    reduction_kernel<<<grid, block, dynamic_shared_memory, queue>>>(num, val);
}

int main()
{
    float* d_A;
    float data[64];
    for (int i = 0; i < 64; i++) {
        data[i] = i / 16;
    }
    cudaMalloc(&d_A, 64 * sizeof(float));
    cudaMemcpy(d_A, data, 64 * sizeof(float), cudaMemcpyHostToDevice);
    reduction_kernel_call(1, 64, 0, GET_QUEUE, 64, d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_A, 64 * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 32; i++) {
        if (i < 16) {
            passed &= (data[i] == 32);
        } else {
            passed &= (data[i] == 64);
        }
    }
    std::cout << "subwarp reduction is " << (passed ? "passed" : "failed")
              << std::endl;
}
