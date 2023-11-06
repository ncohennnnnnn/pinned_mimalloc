#pragma once

#include <cstdlib>

#include <cuda_runtime.h>


template<typename Base>
/* TODO: do we need to know the device id ? Finish this struct anyways */
class user_device_memory: public Base{
public:
    user_device_memory()
    : m_address{nullptr}
    , m_size{0}
    {}

    user_device_memory(void* ptr, const std::size_t size)
    : m_address{ptr}
    , m_size{size}
    {}

    template<typename T>
    user_device_memory(T* ptr, const std::size_t size)
    : m_address{static_cast<void*>(ptr)}
    , m_size{size}
    {
        /* TODO: Check if memory is actually on device first */
    }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

protected:
    void* m_address;
    std::size_t m_size;
    bool m_on_device = true;
};
