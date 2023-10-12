#include "external_allocators/std_alloc.hpp"

#if PMIMALLOC_ALLOCATE_WITH_MPI
#include "external_allocators/MPI_alloc.hpp"
#endif

#if PMIMALLOC_ALLOCATE_WITH_CUDA
#include "external_allocators/cuda_alloc.hpp"
#endif

#if PMIMALLOC_ALLOCATE_WITH_HIP
#include "external_allocators/hip_alloc.hpp"
#endif