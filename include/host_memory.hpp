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

#include <numa.h>
#include <numaif.h>

#include <fmt/core.h>

#if WITH_MIMALLOC
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
#endif


/* TODO: Steal numa stuff from Fabian */
int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((std::size_t)ptr & ~((std::size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("Move page failed from get_node(). \n");
        return -1;
    }
    return numa_node[0];
}

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
    , m_numa_node{0}
    {}

    host_memory(const std::size_t size, const std::size_t alignement = 0) 
    {
        _allocate(m_size, alignement);
    }

    ~host_memory() { _deallocate(); }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

    int get_numa_node(void) { return m_numa_node; }

private:
    void _allocate(const std::size_t size, const std::size_t alignement = 0) {
#if WITH_MIMALLOC
    m_address = std::aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
    if (alignement != 0) { m_address = std::aligned_alloc(alignement, size); }
    else { m_address = std::malloc(size); }
#endif
    m_size = size;
    m_numa_node = get_node(m_address);
}

    void _deallocate() { std::free(m_address); }

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node;
};
