#pragma once

#include <cstdlib>

#if PMIMALLOC_WITH_MIMALLOC
# ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#  define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
# endif
#endif

#include <cuda_runtime.h>

void* aligned_alloc(size_t alignment, size_t size);

template <typename Base>
/* TODO: do we need to know the device id ? Finish this struct anyways */
class device_memory : public Base
{
public:
    device_memory()
      : Base{}
      , m_address{nullptr}
      , m_size{0}
    {
    }

    device_memory(const std::size_t size, const std::size_t alignment = 0)
    {
        _allocate(m_size, alignment);
    }

    template <typename T>
    device_memory(T* ptr, const std::size_t size)
      : Base{}
      , m_address{static_cast<void*>(ptr)}
      , m_size{size}
    {
        /* Check if memory is actually on device first */
        cudaPointerAttributes attributes;
        if (cudaPointerGetAttributes(&attributes, m_address))
        {
            fmt::print(
                "{} : [error] pointer was not allocated or registered by cuda \n", m_address);
            m_address = nullptr, m_size = 0;
        }
    }

    ~device_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
        {
            _deallocate();
        }
        else
        {
            fmt::print("{} : Skipped cudaFree (mi_option_limit_os_alloc) \n", m_address);
        }
    }

    void* get_address(void)
    {
        return m_address;
    }

    std::size_t get_size(void)
    {
        return m_size;
    }

private:
    void _allocate(std::size_t size, std::size_t alignment = 1)
    {
#if PMIMALLOC_WITH_MIMALLOC
        _aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
        if (alignment != 0)
        {
            _aligned_alloc(alignment, size);
        }
        else
        {
            cudaMalloc(&m_address, size);
            m_size = size;
            m_total_size = m_size;
            m_raw_address = m_address;
            fmt::print("{} : Memory of size {} cudaMallocated \n", m_address, size);
        }
#endif
    }

    void _deallocate()
    {
        cudaFree(m_raw_address);
    }

    void _aligned_alloc(size_t alignment, size_t size)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        {
            // Alignment must be a power of 2 and non-zero.
            return;
        }

        // Allocate memory with extra space to store the original pointer.
        size_t total_size = size + alignment - 1;
        m_total_size = total_size;
        cudaMalloc(&m_raw_address, m_total_size);
        fmt::print("{} : Memory of size {} cudaMallocated \n", m_raw_address, m_total_size);

        if (m_raw_address == nullptr)
        {
            return;
        }
        std::cout << std::endl;

        // Calculate the aligned pointer within the allocated memory block.
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(m_raw_address);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;
        // Store the original pointer before the aligned pointer.
        //        uintptr_t* ptr_storage  = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        //        *ptr_storage = reinterpret_cast<uintptr_t>(m_raw_address);

        void* ret = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned pointer \n", ret);

        m_address = ret;
    }

    std::size_t m_total_size;
    void* m_raw_address;

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node = -1;
};
