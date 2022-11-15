#ifndef UNUSED_CUH_
#define UNUSED_CUH_

#include <cooperative_groups.h>


template <typename ValueType, int subwarp_size = 16>
__global__ void reduction_kernel(const int num, ValueType* val)
{
    auto thread_block = cooperative_groups::this_thread_block();
    auto subwarp =
        cooperative_groups::tiled_partition<subwarp_size>(thread_block);
    // block_size only be 64
    __shared__ ValueType tmp[64];
    auto tid = threadIdx.x;
    auto local_data = val[tid];
#pragma unroll
    for (int bitmask = 1; bitmask < subwarp.size(); bitmask <<= 1) {
        const auto remote_data = subwarp.shfl_xor(local_data, bitmask);
        local_data = local_data + remote_data;
    }
    val[tid] = local_data;
}

#endif  // UNUSED_CUH_