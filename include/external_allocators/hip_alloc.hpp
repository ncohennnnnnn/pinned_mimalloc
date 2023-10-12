#pragma once

#include <pmimalloc.hpp>

#include <hip/hip_runtime.h>

/* TODO : Take care of the hipError
*  TODO : Check which flags to select in host allocator
*/
void* hip_alloc(size_t size) { 
    void* rtn = nullptr;
    hipHostMalloc(&rtn, size);
    return rtn;
}
void hip_free(void* ptr) { hipHostFree(ptr); }

ALLOC(hip_alloc, hip_free)