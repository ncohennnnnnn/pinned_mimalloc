#pragma once

#include <cstdlib>

#if WITH_MIMALLOC
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
#endif

#include <cuda_runtime.h>


/* TODO: do we need to know the device id ? Finish this struct anyways */
class device_memory{
public:
    device_memory()
    : m_address{nullptr}
    , m_size{0}
    {}

    device_memory(const std::size_t size, const std::size_t alignement = 0)
    {
        _allocate(m_size, alignement);
    }

    template<typename T>
    device_memory(T* ptr, const std::size_t size)
    : m_address{static_cast<void*>(ptr)}
    , m_size{size}
    {
        /* TODO: Check if memory is actually on device first */
    }

    ~device_memory() { _deallocate(m_address); }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

private:
    void _allocate(std::size_t size, std::size_t alignement = 1) 
    { 
        void* ptr;
        // std::size_t size_buff = 2*size;
        // cudaMalloc(&ptr, size_buff);
        cudaMalloc(&ptr, size);
        /* TODO: probably doesn't work */
// #if WITH_MIMALLOC
//         void* aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size_buff);
// #else
//         void* aligned_ptr = std::align(alignement, size, tmp, size_buff);
// #endif
//         return aligned_ptr; 
        m_address = ptr;
        m_size = size;
    }

    void _deallocate(void* ptr) { cudaFree(ptr); }

protected:
    void* m_address;
    std::size_t m_size;
    bool m_on_device = true;
};
