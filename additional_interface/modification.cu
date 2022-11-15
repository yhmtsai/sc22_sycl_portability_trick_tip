#define GET_QUEUE 0
#include "trick/dim3_t.hpp"


template <typename ValueType>
__global__ void dynamic_shared_memory(ValueType* val)
{
    __shared__ ValueType static_shared[32];
    extern __shared__ ValueType dynamic_shared[];
}

template <typename ValueType>
void dynamic_shared_memory_call(dim3_t grid, dim3_t block,
                   unsigned int dynamic_shared_memsize, cudaStream_t queue,
                   ValueType* val)
{
    dynamic_shared_memory<<<grid, block, dynamic_shared_memsize, queue>>>(val);
}


template <typename ValueType>
__global__ void static_shared_memory(ValueType* val)
{
    __shared__ ValueType static_shared[32];
}

template <typename ValueType>
void static_shared_memory_call(dim3_t grid, dim3_t block,
                   unsigned int dynamic_shared_memsize, cudaStream_t queue,
                   ValueType* val)
{
    static_shared_memory<<<grid, block, dynamic_shared_memsize, queue>>>(val);
}

int main()
{
    double* val;

    // additional call
    dynamic_shared_memory_call(3, 1, 3 * sizeof(double), GET_QUEUE, val);
    static_shared_memory_call(3, 1, 0, GET_QUEUE, val);

    return 0;
}