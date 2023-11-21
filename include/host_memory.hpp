#pragma once

#include <sys/mman.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <errno.h>

#include <numa.hpp>

#include <fmt/core.h>

#if WITH_MIMALLOC
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
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

void* aligned_alloc(size_t alignment, size_t size);

/*TODO: per-thread version where m_size = size * nb_threads */
template<typename Base>
/** @brief Memory living on the host.
 * @fn allocate acts as the body of the constructor.
*/
class host_memory: public Base{
public:
    host_memory()
    : Base{}
    , m_address{nullptr}
    , m_size{0}
    , m_numa_node{-1}
    {}

    host_memory(const std::size_t size, const std::size_t alignement = 0) 
    {
        fmt::print("About to allocate memory of size {} \n", size);
        _allocate(size, alignement);
        fmt::print("{} : Memory std::mallocated\n", m_address);
    }

    ~host_memory()
    { 
        _deallocate(); 
        fmt::print("{} : Memory std::freed\n", m_address); 
    }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

    int get_numa_node(void) { return m_numa_node; }

private:
    void _allocate(const std::size_t size, const std::size_t alignement = 0) {
#if WITH_MIMALLOC
    m_address = _aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
    if (alignement != 0) { m_address = _aligned_alloc(alignement, size); }
    else { m_address = std::malloc(size); }
#endif
    m_size = size;
    // numa_tools n;
    // m_numa_node = numa_tools::get_node(m_address);
    // fmt::print("{} : Pointer is on numa node : {} \n", m_address, m_numa_node);
    m_numa_node = -1;
}

    void _deallocate() { std::free(m_raw_address); }

    void* _aligned_alloc(size_t alignment, size_t size) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            // Alignment must be a power of 2 and non-zero.
            return nullptr;
        }

        // Allocate memory with extra space to store the original pointer.
        size_t total_size = size + alignment - 1;
        void* original_ptr = std::malloc(total_size);
        fmt::print("{} : Raw pointer \n", original_ptr);
        m_raw_address = original_ptr;

        if (original_ptr == nullptr) {
            return nullptr;
        }

        // Calculate the aligned pointer within the allocated memory block.
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(original_ptr);
        uintptr_t misalignment  = unaligned_ptr % alignment;
        uintptr_t adjustment    = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr   = unaligned_ptr + adjustment;
        // Store the original pointer before the aligned pointer.
        uintptr_t* ptr_storage  = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        *ptr_storage = reinterpret_cast<uintptr_t>(original_ptr);

        void* ret = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned pointer \n", ret);

        return ret;
    }

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node;
    void* m_raw_address = nullptr;
};
