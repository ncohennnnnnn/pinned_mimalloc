#include <device.hpp>
#include <pmimalloc.hpp>
#include <log.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#define PMIMALLOC_CHECK_HIP_RESULT(x)                                                               \
    if (x != hipSuccess)                                                                           \
        throw std::runtime_error("pmimalloc error: HIP Call failed " + std::string(#x) + " (" +     \
                                 std::string(hipGetErrorString(x)) + ") in " +                     \
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));

int
get_num_devices()
{
    int n;
    PMIMALLOC_CHECK_HIP_RESULT(hipGetDeviceCount(&n));
    return n;
}

int
get_device_id()
{
    int id;
    PMIMALLOC_CHECK_HIP_RESULT(hipGetDevice(&id));
    return id;
}

void
set_device_id(int id)
{
    PMIMALLOC_CHECK_HIP_RESULT(hipSetDevice(id));
}

void*
device_allocate(std::size_t size)
{
    void* ptr;
    PMIMALLOC_CHECK_HIP_RESULT(hipMalloc(&ptr, size));
    PMIMALLOC_LOG("allocating", size, "bytes using hipMalloc on device", get_device_id(), ":",
        (std::uintptr_t)ptr);
    return ptr;
}

void
device_deallocate(void* ptr) noexcept
{
    PMIMALLOC_LOG("freeing    using hipFree on device", get_device_id(), ":", (std::uintptr_t)ptr);
    hipFree(ptr);
}

void
memcpy_to_device(void* dst, void const* src, std::size_t count)
{
    hipStream_t stream;
    PMIMALLOC_CHECK_HIP_RESULT(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    PMIMALLOC_CHECK_HIP_RESULT(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
    hipEvent_t done;
    PMIMALLOC_CHECK_HIP_RESULT(
        hipEventCreateWithFlags(&done, /*hipEventBlockingSync |*/ hipEventDisableTiming));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventRecord(done, stream));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventSynchronize(done));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventDestroy(done));
}

void
memcpy_to_host(void* dst, void const* src, std::size_t count)
{
    hipStream_t stream;
    PMIMALLOC_CHECK_HIP_RESULT(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    PMIMALLOC_CHECK_HIP_RESULT(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
    hipEvent_t done;
    PMIMALLOC_CHECK_HIP_RESULT(
        hipEventCreateWithFlags(&done, /*hipEventBlockingSync |*/ hipEventDisableTiming));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventRecord(done, stream));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventSynchronize(done));
    PMIMALLOC_CHECK_HIP_RESULT(hipEventDestroy(done));
}
