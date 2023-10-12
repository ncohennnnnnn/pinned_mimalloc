#pragma once

#include <pmimalloc.hpp>

#include <cuda_runtime.h>

/* TODO : Take care of the cudaError
*  TODO : Check which flags to select in host allocator
*/
void* cuda_alloc(size_t size) { 
    void* rtn = nullptr;
    cudaHostAlloc(&rtn, size, cudaHostAllocPortable);
    return rtn;
}
void cuda_free(void* ptr) { cudaFreeHost(ptr); }

ALLOC(cuda_alloc, cuda_free)