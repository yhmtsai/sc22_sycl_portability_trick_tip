#ifndef SYSTEM_HEADER_CUH_
#define SYSTEM_HEADER_CUH_

__device__ void system_with_nd(float* val) {
    auto tid = threadIdx.x;
}


__device__ void system_without_nd(float* val) {
}

#endif // SYSTEM_HEADER_CUH_