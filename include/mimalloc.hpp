#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sys/mman.h>
// #include <sys/types.h>
#include <fmt/core.h>
#include <cstring>
#include <iostream>
#include <unistd.h>

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

// #define MI_MAX_ARENAS         (112)       // not more than 126 (since we use 7 bits in the memid and an arena index + 1)

// typedef struct {
//   mi_arena_id_t id;                       // arena id; 0 for non-specific
//   mi_memid_t memid;                       // memid of the memory area
//   _Atomic(uint8_t*) start;                // the start of the memory area
//   size_t   block_count;                   // size of the area in arena blocks (of `MI_ARENA_BLOCK_SIZE`)
//   size_t   field_count;                   // number of bitmap fields (where `field_count * MI_BITMAP_FIELD_BITS >= block_count`)
//   size_t   meta_size;                     // size of the arena structure itself (including its bitmaps)
//   mi_memid_t meta_memid;                  // memid of the arena structure itself (OS or static allocation)
//   int      numa_node;                     // associated NUMA node
//   bool     exclusive;                     // only allow allocations if specifically for this arena  
//   bool     is_large;                      // memory area consists of large- or huge OS pages (always committed)
//   _Atomic(size_t) search_idx;             // optimization to start the search for free blocks
//   _Atomic(mi_msecs_t) purge_expire;       // expiration time when blocks should be decommitted from `blocks_decommit`.  
//   mi_bitmap_field_t* blocks_dirty;        // are the blocks potentially non-zero?
//   mi_bitmap_field_t* blocks_committed;    // are the blocks committed? (can be NULL for memory that cannot be decommitted)
//   mi_bitmap_field_t* blocks_purge;        // blocks that can be (reset) decommitted. (can be NULL for memory that cannot be (reset) decommitted)  
//   mi_bitmap_field_t  blocks_inuse[1];     // in-place bitmap of in-use blocks (of size `field_count`)
// } mi_arena_t;

// extern mi_decl_cache_align _Atomic(mi_arena_t*) mi_arenas[MI_MAX_ARENAS];

class Mimalloc {
public:
  /**
   * @brief Manages a particular memory arena. Set numa_node to 0 if the node is unknown,
   * set to -1 (or ignore) if no numa node specification is desired.
   */
    Mimalloc(void* addr, const size_t size, const bool is_committed = false,
            const bool is_zero = true, int numa_node = -1) {
        // doesn't consist of large OS pages
        bool is_large = false;

        aligned_size = size;
        // the addr must be 64MB aligned (required by mimalloc)
        if ((reinterpret_cast<uintptr_t>(addr) % MIMALLOC_SEGMENT_ALIGNED_SIZE) != 0) {
            aligned_address = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(addr) +
                                MIMALLOC_SEGMENT_ALIGNED_SIZE - 1) &
                                ~(MIMALLOC_SEGMENT_ALIGNED_SIZE - 1));
            aligned_size = size - ((size_t) aligned_address - (size_t) addr);
        } else {
            aligned_address = addr;
        }

        // Try to pin the allocated memory
        int pin_success = pin(aligned_address, aligned_size, true); // TODO : add error throw

        mi_arena_id_t arena_id;
        bool success = mi_manage_os_memory_ex(aligned_address, aligned_size, is_committed,
                                            is_large, is_zero, numa_node, true, &arena_id);
        if (!success) { // TODO : add error throw
            fmt::print("[error] mimalloc failed to create the arena at {}\n", aligned_address);
            aligned_address = nullptr;
        }
        heap = mi_heap_new_in_arena(arena_id);
        if (heap == nullptr) { // TODO : add error throw
            fmt::print("[error] mimalloc failed to create the heap at {}\n", aligned_address);
            aligned_address = nullptr;
        }

        // do not use OS memory for allocation (but only pre-allocated arena)
        mi_option_set(mi_option_limit_os_alloc, 1);

        // // // get the arena, change its numa_node to the one associated to the heap
        // const size_t arena_index = (size_t)(arena_id <= 0 ? MI_MAX_ARENAS : arena_id - 1); 
        //                             // could use also mi_arena_id_index( arena_id ) instead but no access to the API
        // mi_arena_t* arena = mi_atomic_load_ptr_acquire(mi_arena_t, &mi_arenas[arena_index]);
        // if ( numa_node == 0 ) { arena->numa_node = _mi_os_numa_node_get(&(heap->tld->os)); }
    }

    // leave it undeleted to keep allocated blocks
    ~Mimalloc() {}

    size_t AlignedSize() const { return aligned_size; }

    void* AlignedAddress() const { return aligned_address; }

    void* allocate(const size_t bytes, const size_t alignment = 0) {
        if (unlikely(alignment)) {
            return mi_heap_malloc_aligned(heap, bytes, alignment);
        } else {
            return mi_heap_malloc(heap, bytes);
        }
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
    int pin(void* pointer, size_t bytes, bool pin){ 
        int success;
        std::string str;
        if ( pin ) { 
            success = mlock(&pointer, bytes); 
            str = "pin";
        }
        else { 
            success = munlock(&pointer, bytes); 
            str = "unpin";
        }
        if ( success != 0) { 
            fmt::print("[error] mimalloc failed to {} the allocated memory at {}\n", str, aligned_address);
        }
        return success;
    }

#if ENABLE_DEVICE

    void* allockate(const size_t bytes, const size_t alignment = 0) {
        void* rtn = allocate(bytes, alignment);
        int success = pin(rtn, bytes, true);
        if ( success != 0) { deallocate(rtn); return nullptr;} 
        else { return rtn; }
    }

    void* reallockate(void* pointer, size_t size) {
        size_t bytes = sizeof(pointer);
        int success = pin(pointer, bytes, false);
        if ( success != 0) { return nullptr;}
        void* rtn = mi_heap_realloc(heap, pointer, size);
        success = pin(pointer, size, true);
        if ( success != 0 ) { return nullptr; }
        else { return rtn; }
    }

    void deallockate(void* pointer, size_t size = 0) {
        size_t bytes = sizeof(pointer);
        int success = pin(pointer, bytes, false);
        if ( success != 0) { return;} 
        else { deallocate(pointer, bytes); return; } 
    }

#endif

    size_t getAllocatedSize(void* pointer) { return mi_usable_size(pointer); }

private:
    void* aligned_address = nullptr;
    size_t aligned_size = 0;
    mi_arena_id_t arena_id{};
    mi_heap_t* heap = nullptr;
};


int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((size_t)ptr & ~((size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("move page failed.\n");
        return -1;
    }
    return numa_node[0];
}