#include <device.hpp>
#include <mimalloc.hpp>
#include <log.hpp>
#include <cstdlib>
#include <cstring>

int
get_num_devices()
{
    return 1;
}

int
get_device_id()
{
    return 0;
}

void
set_device_id(int /*id*/)
{
}

void*
device_allocate(std::size_t size)
{
    auto ptr = std::memset(std::malloc(size), 0, size);
    PMIMALLOC_LOG("allocating", size, "bytes using emulate (std::malloc):", (std::uintptr_t)ptr);
    return ptr;
}

void
device_deallocate(void* ptr) noexcept
{
    PMIMALLOC_LOG("freeing    using emulate (std::free):", (std::uintptr_t)ptr);
    std::free(ptr);
}

void
memcpy_to_device(void* dst, void const* src, std::size_t count)
{
    std::memcpy(dst, src, count);
}

void
memcpy_to_host(void* dst, void const* src, std::size_t count)
{
    std::memcpy(dst, src, count);
}

