#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "dpcpp/cooperative_groups.dp.hpp"
#include "dpcpp/dim3.dp.hpp"


template <typename ValueType, int subwarp_size = 16>
void reduction_kernel(const int num, ValueType* val, sycl::nd_item<3> item_ct1,
                      ValueType *tmp)
{
    auto thread_block = this_thread_block(item_ct1);
    auto subwarp = tiled_partition<subwarp_size>(thread_block);
    // block_size only be 64

    auto tid = item_ct1.get_local_id(2);
    auto local_data = val[tid];
#pragma unroll
    for (int bitmask = 1; bitmask < subwarp.size(); bitmask <<= 1) {
        const auto remote_data = subwarp.shfl_xor(local_data, bitmask);
        local_data = local_data + remote_data;
    }
    tmp[tid] = local_data;
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid < 32) {
        tmp[tid] += tmp[tid + 32];
    }
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    val[tid] = tmp[tid];
}

template <typename ValueType, int subwarp_size = 16>
void reduction_kernel_call(dim3 grid, dim3 block,
                           unsigned int dynamic_shared_memory,
                           sycl::queue* queue, const int num, ValueType* val)
{
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_acc_ct1(sycl::range<1>(64), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(subwarp_size)]] {
                             reduction_kernel(
                                 num, val, item_ct1,
                                 (ValueType*)tmp_acc_ct1.get_pointer());
                         });
    });
}

int main()
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    float* d_A;
    float data[64];
    for (int i = 0; i < 64; i++) {
        data[i] = i / 16;
    }
    d_A = sycl::malloc_device<float>(64, q_ct1);
    q_ct1.memcpy(d_A, data, 64 * sizeof(float)).wait();
    reduction_kernel_call(1, 64, 0, &q_ct1, 64, d_A);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(data, d_A, 64 * sizeof(float)).wait();
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