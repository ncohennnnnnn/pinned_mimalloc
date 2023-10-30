#pragma once

#include <pmimalloc.hpp>

#include <cuda_runtime.h>

/* TODO : Take care of the cudaError
*  TODO : Check which flags to select in host allocator
*/
void* cuda_alloc(size_t size) { 
    void* rtn;
    cudaError_t cudaStatus = cudaHostAlloc((void**)&rtn, size, cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) {
        fmt::print("cudaHostAlloc failed: {} \n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }
    return rtn;
}
void cuda_free(void* ptr) { 
    cudaError_t cudaStatus = cudaFreeHost(ptr);
    if (cudaStatus != cudaSuccess) {
        fmt::print("cudaFreeHost failed: {} \n", cudaGetErrorString(cudaStatus));
    }
}

ALLOC(cuda_alloc, cuda_free)