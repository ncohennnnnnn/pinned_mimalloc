#pragma once

#include <cstdint>
#define EXT_HEAP_SZ_EXP 25

uint32_t ext_heap_sz = 1 << 25;


// #define PMIMALLOC_ENABLE_DEVICE 1
#define PMIMALLOC_DEVICE_RUNTIME "CUDA"
#define PMIMALLOC_DEVICE_CUDA
// #define PMIMALLOC_ENABLE_LOGGING
