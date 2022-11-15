#include "local.cuh"
#include "system_header.cuh"
__global__ void dummy(float* val) {
    system_with_nd(val);
    system_without_nd(val);
    local_with_nd(val);
    local_without_nd(val);
}

void dummy_call(float *val) {
    dummy<<<1, 1>>>(val);
}