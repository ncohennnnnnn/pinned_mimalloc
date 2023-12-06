#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/core.h>

#if PMIMALLOC_WITH_MIMALLOC
# ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#  define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
# endif
#endif

#include <pmimalloc/host_memory.hpp>

template <typename Base>
/** @brief Already allocated memory living on the host. */
class user_host_memory : public Base
{
public:
    user_host_memory()
      : Base{}
      , m_address{nullptr}
      , m_size{0}
      , m_numa_node{-1}
    {
    }

    user_host_memory(void* ptr, const std::size_t size)
      : Base{}
      , m_address{ptr}
      , m_size{size}
    {
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
        m_numa_node = -1;
    }

    ~user_host_memory() {}

    void destroy()
    {
        _deallocate();
    }

    void* get_address(void)
    {
        return m_address;
    }

    std::size_t get_size(void)
    {
        return m_size;
    }

    int get_numa_node(void)
    {
        return m_numa_node;
    }

private:
    void _deallocate()
    {
        std::free(m_address);
    }

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node;
};
