#include <system_header.cuh>


__device__ void local_with_nd(float* val) {
    auto tid = threadIdx.x;
    system_with_nd(val);
}


__device__ void local_without_nd(float* val) {
    system_without_nd(val);
}