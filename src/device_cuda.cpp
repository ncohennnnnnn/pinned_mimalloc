#include <device.hpp>
#include <mimalloc.hpp>
#include <log.hpp>

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define PMIMALLOC_CHECK_CUDA_RESULT(x)                                                              \
    if (x != cudaSuccess)                                                                          \
        throw std::runtime_error("pinned_mimalloc error: CUDA Call failed " + std::string(#x) + " (" +    \
                                 std::string(cudaGetErrorString(x)) + ") in " +                    \
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));


int
get_num_devices()
{
    int n;
    PMIMALLOC_CHECK_CUDA_RESULT(cudaGetDeviceCount(&n));
    return n;
}

int
get_device_id()
{
    int id;
    PMIMALLOC_CHECK_CUDA_RESULT(cudaGetDevice(&id));
    return id;
}

void
set_device_id(int id)
{
    PMIMALLOC_CHECK_CUDA_RESULT(cudaSetDevice(id));
}

void*
device_malloc(std::size_t size)
{
    void* ptr;
    PMIMALLOC_CHECK_CUDA_RESULT(cudaMalloc(&ptr, size));
    PMIMALLOC_LOG("allocating", size, "bytes using cudaMalloc on device", get_device_id(), ":",
        (std::uintptr_t)ptr);
    return ptr;
}

void
device_free(void* ptr) noexcept
{
    PMIMALLOC_LOG("freeing    using cudaFree on device", get_device_id(), ":", (std::uintptr_t)ptr);
    cudaFree(ptr);
}

void
memcpy_to_device(void* dst, void const* src, std::size_t count)
{
    cudaStream_t stream;
    PMIMALLOC_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
    cudaEvent_t done;
    PMIMALLOC_CHECK_CUDA_RESULT(
        cudaEventCreateWithFlags(&done, /*cudaEventBlockingSync |*/ cudaEventDisableTiming));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventRecord(done, stream));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventSynchronize(done));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventDestroy(done));
}

void
memcpy_to_host(void* dst, void const* src, std::size_t count)
{
    cudaStream_t stream;
    PMIMALLOC_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
    cudaEvent_t done;
    PMIMALLOC_CHECK_CUDA_RESULT(
        cudaEventCreateWithFlags(&done, /*cudaEventBlockingSync |*/ cudaEventDisableTiming));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventRecord(done, stream));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventSynchronize(done));
    PMIMALLOC_CHECK_CUDA_RESULT(cudaEventDestroy(done));
}

