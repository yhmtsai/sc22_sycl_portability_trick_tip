template<typename ValueType>
__global__ void dynamic_shared_memory(ValueType* val) {
    __shared__ ValueType static_shared[32];
    extern __shared__ ValueType dynamic_shared[];
}


template<typename ValueType>
__global__ void static_shared_memory(ValueType* val) {
    __shared__ ValueType static_shared[32];
}


template<typename ValueType>
void call(ValueType* val) {
     // direct call
    dynamic_shared_memory<<<3, 1, 3*sizeof(double)>>>(val);
    static_shared_memory<<<3, 1>>>(val);
}


int main() {
    double* val;

    // direct call
    dynamic_shared_memory<<<3, 1, 3*sizeof(double)>>>(val);
    static_shared_memory<<<3, 1>>>(val);

    return 0;
}