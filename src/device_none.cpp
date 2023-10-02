#include <device.hpp>
#include <mimalloc.hpp>


int
get_num_devices()
{
    return 0;
}

int
get_device_id()
{
    return 0;
}

void
set_device_id(int)
{
}

void*
device_allocate(std::size_t)
{
    return nullptr;
}

void
device_deallocate(void*) noexcept
{
}

void
memcpy_to_device(void*, void const*, std::size_t)
{
}

void
memcpy_to_host(void*, void const*, std::size_t)
{
}
