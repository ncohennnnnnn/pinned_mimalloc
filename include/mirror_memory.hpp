#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#include <numa.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

#if PMIMALLOC_WITH_MIMALLOC
# ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#  define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
# endif
#endif

// if we allocate memory using regular malloc/std::malloc when mimalloc has overridden
// the default allocation, then we end up creating our heap with a chunk of memory that
// came from mimalloc. Instead we must use mmap/munmap to get system memory that
// isn't part of the mimalloc heap tracking/usage
#define PMIMALLOC_USE_MMAP

template <typename Base>
/** @brief Memory mirrored on the host and device. 
 * TODO:  do we need to know the device id ?
*/
class mirror_memory : public Base
{
public:
    mirror_memory() = default;

    mirror_memory(const std::size_t size, const std::size_t alignment = 0)
      : Base{}
    {
        _allocate(size, alignment);
    }

    ~mirror_memory()
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
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", m_raw_address);
            cudaFree(m_raw_address_device);
            fmt::print("{} : Device memory cudaFreed \n", m_raw_address_device);
        }
    }

    void* get_address(void)
    {
        return m_address;
    }

    void* get_address_device(void)
    {
        return m_address_device;
    }

    std::size_t get_size(void)
    {
        return m_size;
    }

    int get_numa_node(void)
    {
        return m_numa_node;
    }

private:
    void _allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        m_size = size;
#if PMIMALLOC_WITH_MIMALLOC
        _aligned_alloc_mirror(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
        if (alignment != 0)
        {
            _aligned_alloc_mirror(alignment, size);
        }
        else
        {
            m_address = std::malloc(m_size);
            m_raw_address = m_address;
            m_total_size = m_size;
            fmt::print("{} : Memory of size {} std::mallocated \n", m_address_device, m_size);

            cudaMalloc(&m_address_device, m_size);
            m_raw_address_device = m_address_device;
            m_total_size = m_size;
            fmt::print("{} : Memory of size {} cudaMallocated \n", m_address_device, m_size);
        }
#endif
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
        // fmt::print("{} : Pointer is on numa node : {} \n", m_address, m_numa_node);
        m_numa_node = -1;
    }

    void _deallocate()
    {
        cudaFree(m_raw_address_device);
        fmt::print("{} : Device memory cudaFreed \n", m_raw_address_device);
#ifdef PMIMALLOC_USE_MMAP
        if (munmap(m_raw_address, m_total_size) != 0)
        {
            fmt::print("{} : [error] munmap failed \n", m_raw_address);
            return;
        }
        fmt::print("{} : Host memory munmaped \n", m_raw_address);
#else
        std::free(m_raw_address);
        fmt::print("{} : Host memory std::freed \n", m_raw_address);
#endif
    }

    void _aligned_alloc_mirror(const std::size_t alignment, size_t size)
    {
        /* Check the alignment, must be a power of 2 and non-zero. */
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            return;

        size_t total_size = size + alignment - 1;
        m_total_size = total_size;

        /* Allocate on host. */
        _aligned_host_alloc(alignment);

        /* Allocate on device */
        _aligned_device_alloc(alignment);
    }

    void _aligned_host_alloc(const std::size_t alignment)
    {
#ifdef PMIMALLOC_USE_MMAP
        void* original_ptr =
            mmap(0, m_total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (original_ptr == MAP_FAILED)
        {
            std::cerr << "[error] mmap failed (error thrown) \n" << std::endl;
            original_ptr = nullptr;
            return;
        }
        else
        {
            /* Tell the OS that this memory can be considered for huge pages */
            madvise(original_ptr, m_total_size, MADV_HUGEPAGE);
        }
        fmt::print("{} : Host memory of size {} mmaped \n", original_ptr, m_total_size);
#else
        void* original_ptr = std::malloc(m_total_size);
        fmt::print("{} : Host memory of size {} std::mallocated \n", original_ptr, m_total_size);
#endif
        if (original_ptr == nullptr)
        {
            fmt::print("[error] Host allocation failed (nullptr) \n");
            return;
        }

        m_raw_address = original_ptr;

        /* Calculate the aligned pointer within the allocated memory block. */
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(m_raw_address);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;

        m_address = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned host pointer \n", m_address);
    }

    void _aligned_device_alloc(const std::size_t alignment)
    {
        cudaMalloc(&m_raw_address_device, m_total_size);
        fmt::print(
            "{} : Device memory of size {} cudaMallocated \n", m_raw_address_device, m_total_size);

        if (m_raw_address_device == nullptr)
            return;

        /* Calculate the aligned pointer within the allocated memory block. */
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(m_raw_address_device);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;

        void* rtn = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned device pointer \n", rtn);

        m_address_device = rtn;
    }

protected:
    void* m_address;
    void* m_address_device;
    void* m_raw_address;
    void* m_raw_address_device;
    std::size_t m_size;
    std::size_t m_total_size;
    int m_numa_node;
};

// template <typename T>
// mirror_memory(T* ptr, const std::size_t size, const std::size_t /*alignment*/ = 0)
//   : Base{}
//   , m_address{static_cast<void*>(ptr)}
//   , m_raw_address{}
//   , m_size{size}
// {
//     /* Check if memory is actually on device first */
//     cudaPointerAttributes attributes;
//     if (cudaPointerGetAttributes(&attributes, m_address))
//     {
//         /* Check the alignment, must be a power of 2 and non-zero. */
//         if (alignment == 0 || (alignment & (alignment - 1)) != 0)
//             fmt::print("{} : [error] Wrong alignment \n", ptr);

//         size_t total_size = size + alignment - 1;
//         m_total_size = total_size;
//     }
// }