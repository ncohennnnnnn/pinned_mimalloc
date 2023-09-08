#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <errno.h>

#include <fmt/core.h>

#include <numa.h>
#include <numaif.h>

#include <mimalloc.h>
#include <mimalloc/atomic.h>
#include <mimalloc/internal.h>
#include <../src/bitmap.h>

#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__IBMC__) || \
    defined(__INTEL_COMPILER) || defined(__clang__)
#ifndef unlikely
#define unlikely(x_) __builtin_expect(!!(x_), 0)
#endif
#ifndef likely
#define likely(x_) __builtin_expect(!!(x_), 1)
#endif
#else
#ifndef unlikely
#define unlikely(x_) (x_)
#endif
#ifndef likely
#define likely(x_) (x_)
#endif
#endif

#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif


/**
* @brief The address must be 64MB aligned (required by mimalloc).
* One can use mmap instead of alloc + align, the latter is safer but recquires
* to allocate more memory than needed since alignement will waste some.
*/
#define ALIGNED(allocator, deallocator) \
void* allocator##_aligned(size_t size){ \
    if ( #allocator == "std_malloc" ) { return aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size); } \
    void* tmp = allocator(size); \
    fmt::print("Raw pointer is at {} \n", tmp); \
    void* aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size); \
    fmt::print("Aligned pointer is at {} \n", aligned_ptr); \
    return aligned_ptr; \
} \


int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((size_t)ptr & ~((size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("Move page failed.\n");
        return -1;
    }
    return numa_node[0];
}


class Mimalloc {
public:
  /**
   * @brief Manages a particular memory arena. 
   * numa_node to 0 if no numa node, ignore if unknown.
   */
    Mimalloc(void* addr, const size_t size, const bool is_committed = false,
            const bool is_zero = true, int numa_node = -1) {
        // doesn't consist of large OS pages
        bool is_large = false;

        aligned_size = size;
        aligned_address = addr;

        // Find NUMA node if not known before 
        if ( numa_node == -1 ) { numa_node = get_node(aligned_address); }

        bool success = mi_manage_os_memory_ex(aligned_address, aligned_size, is_committed,
                                            is_large, is_zero, numa_node, true, &arena_id);
        if (!success) { // TODO : add error throw
            fmt::print("[error] mimalloc failed to create the arena at {} \n", aligned_address);
            aligned_address = nullptr;
        }
        heap = mi_heap_new_in_arena(arena_id);
        if (heap == nullptr) { // TODO : add error throw
            fmt::print("[error] mimalloc failed to create the heap at {} \n", aligned_address);
            aligned_address = nullptr;
        }

        // do not use OS memory for allocation (but only pre-allocated arena)
        mi_option_set(mi_option_limit_os_alloc, 1);

        // Pin the allocated memory
        int pin_success = pin(true); // TODO : add error throw
    }

    // leave it undeleted to keep allocated blocks
    ~Mimalloc() { 
        int success = pin(false);
    }

    size_t AlignedSize() const { return aligned_size; }

    void* AlignedAddress() const { return aligned_address; }

    void* allocate(const size_t bytes, const size_t alignment = 0) {
        void* rtn = nullptr;
        if (unlikely(alignment)) {
            rtn = mi_heap_malloc_aligned(heap, bytes, alignment);
        } else {
            rtn = mi_heap_malloc(heap, bytes);
        }
        fmt::print("memory allocated at {}\n", rtn);
        return rtn;
    }

    void* reallocate(void* pointer, size_t size) {
        return mi_heap_realloc(heap, pointer, size);
    }

    void deallocate(void* pointer, size_t size = 0) {
        if (likely(pointer)) {
            if (unlikely(size)) { mi_free_size(pointer, size); }
            else { mi_free(pointer); }
        }
    }

    /**
    * @brief Set bool pin to true to pin the memory, false to unpin it
    */ 
    int pin(bool pin){ 
        int success;
        std::string str;
        if ( pin ) { 
            success = mlock(&aligned_address, aligned_size); 
            str = "pin";
        }
        else { 
            success = munlock(&aligned_address, aligned_size); 
            str = "unpin";
        }
        if ( success != 0) { 
            fmt::print("[error] mimalloc failed to {} the allocated memory at {} : ", str, aligned_address);
            if        (errno == EAGAIN) {
                fmt::print("EAGAIN. \n (Some or all of the specified address range could not be locked.) \n");
            } else if (errno == EINVAL) {
                fmt::print("EINVAL. \n (The result of the addition addr+len was less than addr. addr = {} and len = {})\n", aligned_address, aligned_size);
            } else if (errno == ENOMEM) {
                fmt::print("ENOMEM. \n (Some of the specified address range does not correspond to mapped pages in the address space of the process.) \n");
            } else if (errno == EPERM ) {
                fmt::print("EPERM. \n (The caller was not privileged.) \n");
            }
        }
        return success;
    }

    size_t getAllocatedSize(void* pointer) { return mi_usable_size(pointer); }

private:
    void* aligned_address = nullptr;
    size_t aligned_size = 0;
    mi_arena_id_t arena_id{};
    mi_heap_t* heap = nullptr;
    mi_stats_t stats;
};
