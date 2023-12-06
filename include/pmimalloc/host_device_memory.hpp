#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
//
#include <cuda_runtime.h>
#include <fmt/core.h>
//
#include <pmimalloc/numa.hpp>

#if PMIMALLOC_WITH_MIMALLOC
# ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#  define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
# endif
#endif

// if we allocate memory using regular malloc/std::malloc when mimalloc has overridden
// the default allocation, then we end up creating our heap with a chunk of memory that
// came from mimalloc. Instead we must use mmap/munmap to get system memory that
// isn't part of the mimalloc heap tracking/usage

template <typename Base>
/** @brief Memory mirrored on the host and device.
 * TODO:  do we need to know the device id ?
*/
class host_device_memory : public Base
{
public:
    host_device_memory() = default;

    host_device_memory(const std::size_t size, const std::size_t alignment = 0)
      : Base{}
      , m_size{size}
      , m_numa_node{-1}
    {
#if PMIMALLOC_WITH_MIMALLOC
        _mirror_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
        _mirror_alloc(alignment, size);
#endif
    }

    ~host_device_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
            _host_dealloc();
        else
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", m_raw_address);

        _device_dealloc();
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
    void _host_alloc(const std::size_t alignment = 0)
    {
        _set_total_size(alignment);

        m_raw_address =
            mmap(0, m_total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m_raw_address == MAP_FAILED)
        {
            std::cerr << "[error] mmap failed (error thrown) \n" << std::endl;
            m_raw_address = nullptr;
            return;
        }
        else
        {
            /* Tell the OS that this memory can be considered for huge pages */
            madvise(m_raw_address, m_total_size, MADV_HUGEPAGE);
        }
        fmt::print("{} : Host memory of size {} mmaped \n", m_raw_address, m_total_size);

        if (m_raw_address == nullptr)
        {
            fmt::print("[error] Host allocation failed \n");
            return;
        }

        m_address = _align(m_raw_address, alignment);
        fmt::print("{} : Aligned host pointer \n", m_address);
    }

    void _device_alloc(const std::size_t alignment = 0)
    {
        _set_total_size(alignment);

        cudaMalloc(&m_raw_address_device, m_total_size);
        fmt::print(
            "{} : Device memory of size {} cudaMallocated \n", m_raw_address_device, m_total_size);

        if (m_raw_address_device == nullptr)
        {
            fmt::print("[error] Device allocation failed \n");
            return;
        }

        m_address_device = _align(m_raw_address_device, alignment);
        fmt::print("{} : Aligned device pointer \n", m_address_device);
    }

    void _mirror_alloc(const std::size_t alignment, size_t size)
    {
        _set_total_size(alignment);
        _host_alloc(alignment);
        _device_alloc(alignment);
    }

    void _host_dealloc(void)
    {
        if (munmap(m_raw_address, m_total_size) != 0)
        {
            fmt::print("{} : [error] munmap failed \n", m_raw_address);
            return;
        }
        fmt::print("{} : Host memory munmaped \n", m_raw_address);
    }

    void _device_dealloc(void)
    {
        cudaFree(m_raw_address_device);
        fmt::print("{} : Device memory cudaFreed \n", m_raw_address_device);
    }

    void _set_total_size(const std::size_t alignment)
    {
        if (alignment == 0)
        {
            m_total_size = m_size;
            return;
        }
        else if ((alignment & (alignment - 1)) != 0)
        {
            fmt::print("[error] null or odd alignement! \n");
            return;
        }

        m_total_size = m_size + alignment - 1;
    }

    /* Calculate the aligned pointer within the allocated memory block. */
    void* _align(void* ptr, const std::size_t alignment)
    {
        if (alignment == 0)
            return ptr;
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;
        void* rtn = reinterpret_cast<void*>(aligned_ptr);
        return rtn;
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