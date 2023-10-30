#pragme once

#include <sys/mman.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>

#if WITH_MIMALLOC
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
#endif

#include <fmt/core.h>

/** @brief Memory living on the host.
 * @fn allocate_aligned and @fn allocate act as constructors.
*/
struct host_memory{
    
    /** @returns the associated numa node. */
    int allocate(const std::size_t size, const std::size_t alignement = 0) {
        m_size = size;
#if WITH_MIMALLOC
        m_address = std::aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, m_size);
#else
        if (alignement != 0) { m_address = std::aligned_alloc(alignement, m_size); }
        else { m_address = std::malloc(m_size); }
#endif
        return get_node(m_address);
    }

    void deallocate() { std::free(m_address); }

    bool pin_or_unpin(void* ptr, const size_t size, bool pin){
        int success;
        std::string str;
        if (pin) {
            success = mlock(ptr, size);
            str = "pin";
        } else { 
            success = munlock(ptr, size);
            str = "unpin";
        }
        if (success != 0) { 
            fmt::print("{} : [error] failed to {} the allocated memory : ", ptr, str);
            if        (errno == EAGAIN) {
                fmt::print("EAGAIN. \n (Some or all of the specified address range could not be locked.) \n");
            } else if (errno == EINVAL) {
                fmt::print("EINVAL. \n (The result of the addition addr+len was less than addr. addr = {} and len = {})\n", ptr, size);
            } else if (errno == ENOMEM) {
                fmt::print("ENOMEM. \n (Some of the specified address range does not correspond to mapped pages in the address space of the process.) \n");
            } else if (errno == EPERM ) {
                fmt::print("EPERM. \n (The caller was not privileged.) \n");
            }
        } else { fmt::print("{} : Memory {}ned. \n", ptr, str); } 
        return (bool)(1-success);
    }

    void* get_address(void) { return m_address; }

    std::size_t get_size(void) { return m_size; }

private:
    std::size_t m_size;
    void* m_address;
};

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