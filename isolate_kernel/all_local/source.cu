#include "unused.cuh"

template <typename ValueType>
__global__ void dummy(int num, ValueType* val) {
    auto tid = get_index();
    val[threadIdx.x] += 1;
    
}

int main () {
    float* d_A;
    float data[64];
    for (int i = 0; i < 64; i++) {
        // 0~15: 0, 16~31: 1
        data[i] = i / 16;
    }
    cudaMalloc(&d_A, 64 * sizeof(float));
    cudaMemcpy(d_A, data, 64 * sizeof(float), cudaMemcpyHostToDevice);
    dummy<<<1, 64>>>(64, d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_A, 64 * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}