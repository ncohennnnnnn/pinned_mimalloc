#pragma once

#include <pmimalloc.hpp>

#define std_alloc(x)    std::malloc(x)
#define std_free(x)     std::free(x)

ALLOC(std_alloc, std_free)