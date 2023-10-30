#pragma once

#include <cstdlib>

#if WITH_MIMALLOC
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
#endif

#include <fmt/core.h>

#include <cuda_runtime.h>

/* TODO: do we need to know the device id ? Finish this struct anyways*/
struct device_memory{
    void* allocate_aligned(std::size_t size, std::size_t alignement = 1) { 
        std::size_t size_buff = 2*size;
        void* tmp = cudaMalloc(size_buff);
        /* TODO: probably doesn't work */
#if WITH_MIMALLOC
        void* aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size_buff);
#else
        void* aligned_ptr = std::align(alignement, size, tmp, size_buff);
#endif
        return aligned_ptr; 
    }

    void deallocate(void* ptr) { return cudaFree(ptr); }

    void pin_or_unpin(void* ptr, const size_t size, bool pin){
        cudaError_t cudaStatus;
        if (pin) {
            /* TODO: Choose the appropriate flags */
            cudaStatus = cudaHostRegister( ptr, size, cudaHostRegisterDefault ) ;
            if (cudaStatus != cudaSuccess) {
                fmt::print("cudaHostRegister failed: {} \n", cudaGetErrorString(cudaStatus));
            } else { fmt::print("{} : Memory pinned (by CUDA). \n", ptr); }
        } else {
            /* TODO: Choose the appropriate flags */
            cudaStatus = cudaHostUnregister( ptr ) ;
            if (cudaStatus != cudaSuccess) {
                fmt::print("cudaHostUnregister failed: {} \n", cudaGetErrorString(cudaStatus));
            } else { fmt::print("{} : Memory unpinned (by CUDA). \n", ptr); }
        }
    }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

private:
    std::size_t m_size;
    void* m_address;
};
