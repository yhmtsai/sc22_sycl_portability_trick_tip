#include <dpct/dpct.hpp>

template <unsigned Size>
class thread_block_tile {
    using sub_group = sycl::sub_group;
    using id_type = sub_group::id_type;

public:
    // note: intel calls nd_item.get_sub_group(), but it still call
    // sycl::sub_group() to create the sub_group.
    template <typename Group>
    explicit thread_block_tile(const Group& parent_group)
        : data_{Size, 0}, sub_(parent_group.get_sub_group())
    {
#ifndef NDEBUG
        assert(sub_.get_local_range().get(0) == Size);
#endif
        data_.rank = sub_.get_local_id();
    }


    __dpct_inline__ unsigned thread_rank() const noexcept { return data_.rank; }

    __dpct_inline__ unsigned size() const noexcept { return Size; }

    __dpct_inline__ void sync() const noexcept { sub_.barrier(); }

#define GKO_BIND_SHFL(ShflOpName, ShflOp)                                      \
    template <typename ValueType, typename SelectorType>                       \
    __dpct_inline__ ValueType ShflOpName(ValueType var, SelectorType selector) \
        const noexcept                                                         \
    {                                                                          \
        return sub_.ShflOp(var, selector);                                    \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

    GKO_BIND_SHFL(shfl_xor, shuffle_xor);


private:
    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
    sub_group sub_;
};


__dpct_inline__ sycl::nd_item<3> this_thread_block(sycl::nd_item<3>& item_ct1) {
    return item_ct1;
}

template <int subgroup_size>
using tiled_partition = thread_block_tile<subgroup_size>;