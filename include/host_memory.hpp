#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#include <numa.hpp>

#include <fmt/core.h>

#if WITH_MIMALLOC
# ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#  define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
# endif
#endif

// /* TODO: Steal numa stuff from Fabian */
// int get_node(void* ptr){
//     int numa_node[1] = {-1};
//     void* page = (void*)((std::size_t)ptr & ~((std::size_t)getpagesize()-1));
//     int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
//     if (err == -1) {
//         fmt::print("Move page failed from get_node(). \n");
//         return -1;
//     }
//     return numa_node[0];
// }

/*TODO: per-thread version where m_size = size * nb_threads */
template <typename Base>
/** @brief Memory living on the host.
 * @fn allocate acts as the body of the constructor.
*/
class host_memory : public Base
{
public:
    host_memory()
      : Base{}
      , m_address{nullptr}
      , m_size{0}
      , m_numa_node{-1}
    {
    }

    host_memory(const std::size_t size, const std::size_t alignement = 0)
    {
        _allocate(size, alignement);
        // fmt::print("Host memory created with m_address : {}, m_size : {}, m_raw_address : {}, m_total_size : {} \n"
        //     , m_address, m_size, m_raw_address, m_total_size);
    }

    ~host_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
        {
            _deallocate();
        }
        else
        {
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", m_address);
        }
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
    void _allocate(const std::size_t size, const std::size_t alignement = 0)
    {
#if WITH_MIMALLOC
        _aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
        if (alignement != 0)
        {
            _aligned_alloc(alignement, size);
        }
        else
        {
            m_address = std::malloc(size);
            m_raw_address = m_address;
            m_total_size = size;
            fmt::print("{} : Memory of size {} std::mallocated \n", m_address, size);
        }
#endif
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
        // fmt::print("{} : Pointer is on numa node : {} \n", m_address, m_numa_node);
        m_numa_node = -1;
        m_size = size;
    }

#define PMIMALLOC_USE_MMAP
    void _deallocate()
    {
#ifdef PMIMALLOC_USE_MMAP
        if (munmap(m_raw_address, m_total_size) != 0)
        {
            std::cerr << "munmap failed \n" << std::endl;
        }
        fmt::print("{} : Memory munmaped \n", m_raw_address);
#else
        std::free(m_raw_address);
        fmt::print("{} : Memory std::freed \n", m_raw_address);
#endif
    }

    void _aligned_alloc(size_t alignment, size_t size)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        {
            // Alignment must be a power of 2 and non-zero.
            return;
        }

        // Allocate memory with extra space to store the original pointer.
        size_t total_size = size + alignment - 1;
#ifdef PMIMALLOC_USE_MMAP
        void* original_ptr =
            mmap(0, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (original_ptr == MAP_FAILED)
        {
            std::cerr << "[error] mmap failed (error thrown) \n" << std::endl;
            original_ptr = nullptr;
        }
        fmt::print("{} : Memory of size {} mmaped \n", original_ptr, total_size);
#else
        void* original_ptr = std::malloc(total_size);
        fmt::print("{} : Memory of size {} std::mallocated \n", original_ptr, total_size);
#endif
        m_raw_address = original_ptr;
        m_total_size = total_size;

        if (original_ptr == nullptr)
        {
            fmt::print("[error] mmap failed (nullptr) \n");
            return;
        }

        // Calculate the aligned pointer within the allocated memory block.
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(m_raw_address);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;
        // Store the original pointer before the aligned pointer.
        //        uintptr_t* ptr_storage  = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        //        *ptr_storage = reinterpret_cast<uintptr_t>(original_ptr);

        m_address = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned pointer \n", m_address);
        // m_size    = size;
    }

    // std::size_t m_total_size;
    // void*       m_raw_address;

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node;
    std::size_t m_total_size;
    void* m_raw_address;
};
