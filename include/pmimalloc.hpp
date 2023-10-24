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

#include <mpi.h>

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

/** @brief The address must be 64MB aligned (required by mimalloc).
*   @brief Creates an aligned allocator and a deallocator that unpins first.
* TODO : Find a way to somehow not lose half the allocated memory
*/
#define ALLOC(allocator, deallocator) \
void* ext_##allocator##_aligned(std::size_t size){ \
    void* aligned_ptr; \
    if ( #allocator == "std_alloc") { aligned_ptr = aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size); } \
    else { \
        std::size_t size_buff = 2*size; \
        void* tmp = allocator(size_buff); \
        aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size_buff); \
    } \
    fmt::print("{} : Aligned {}ated ptr. \n", aligned_ptr, #allocator); \
    return aligned_ptr; \
} \
void ext_##deallocator(pmimalloc mim){ \
    mim.unpin(); \
    mim.win_free(); \
    deallocator(mim.get_Address()); \
} \

int get_node(void* ptr);

class pmimalloc {

    /** @param pin true to pin the arena, false to unpin it. */ 
    void pin_or_unpin(bool pin);

public:
  /** @brief Creates a mimalloc arena from externally allocated memory. 
   * @param addr is the adress of the chunk of memory.
   * @param numa_node 0 if single numa node, ignore if unknown.
   */
    pmimalloc(void* addr, const std::size_t size, const bool device = false, const int numa_node = -1);

    /* Leave it undeleted to keep allocated blocks */
    ~pmimalloc() {}

    void* allocate(const std::size_t bytes, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

    std::size_t get_UsableSize(void* ptr) { return mi_usable_size(ptr); }

    std::size_t get_TotalSize() const { return m_size; }

    void* get_Address() const { return m_address; }

    /* Pins the whole arena. */
    void pin();

    /* Unpins the whole arena. */
    void unpin();

    /* Frees the MPI window associated to the arena. */
    void win_free();

private:
    void* m_address = nullptr;
    bool m_device_allocated = false;
    int m_numa_node = 0;
    MPI_Win m_win;
    std::size_t m_size = 0;
    mi_arena_id_t m_arena_id{};
    mi_heap_t* m_heap = nullptr;
    mi_stats_t m_stats;
};
