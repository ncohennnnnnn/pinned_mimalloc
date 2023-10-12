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

/**
* @brief The address must be 64MB aligned (required by mimalloc).
* Creates an aligned allocator and a deallocator that unpins first.
* /!\ Change "std_alloc_TEST" back to "std_alloc" /!\
*
* TODO : Find a way to somehow not lose half the allocated memory
*/
#define ALLOC(allocator, deallocator) \
void* ext_##allocator##_aligned(std::size_t size){ \
    if ( #allocator == "std_alloc_TEST" ) { return aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size); } \
    std::size_t size_buff = 2*size; \
    void* tmp = allocator(size_buff); \
    fmt::print("{} : Raw {}ated ptr. \n", tmp, #allocator); \
    void* aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size_buff); \
    fmt::print("{} : Aligned {}ated ptr. \n", aligned_ptr, #allocator); \
    return aligned_ptr; \
} \
void ext_##deallocator(pmimalloc mim){ \
    mim.unpin(); \
    deallocator(mim.AlignedAddress()); \
} \

// // Define a type trait to map C++ types to MPI_Datatypes
// template <typename T>
// struct MpiTypeMapper {
//     static MPI_Datatype get() { return MPI_BYTE; } // Default when unsupported type
// };

// template <>
// struct MpiTypeMapper<int> {
//     static MPI_Datatype get() { return MPI_INT; }
// };

// template <>
// struct MpiTypeMapper<double> {
//     static MPI_Datatype get() { return MPI_DOUBLE; }
// };

int get_node(void* ptr);

class pmimalloc {
public:
  /**
   * @brief Creates a mimalloc arena from externally allocated memory. 
   * @param addr is the adress of the chunk of memory.
   * @param size its size.
   * @param is_committed
   * @param is_zero
   * @param numa_node 0 if single numa node, ignore if unknown.
   */
    pmimalloc(void* addr, const std::size_t size, const bool is_committed = true,
            const bool is_zero = false, int numa_node = -1, const bool device = false);

    /* Leave it undeleted to keep allocated blocks */
    ~pmimalloc() {}

    std::size_t AlignedSize() const { return aligned_size; }

    void* AlignedAddress() const { return aligned_address; }

    void* allocate(const std::size_t bytes, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

    std::size_t getAllocatedSize(void* ptr) { return mi_usable_size(ptr); }

    void* device_allocate(std::size_t size);

    // void* device_reallocate(void* ptr, std::size_t size);

    void device_deallocate(void* ptr) noexcept;
    
private:
    /** 
     * @param pin true to pin the arena, false to unpin it. 
     */ 
    int pin_or_unpin(bool pin);

    void pin(){ int success = pin_or_unpin(true); }

    void unpin(){ int success = pin_or_unpin(false); }

private:
    void* aligned_address = nullptr;
    // int key;
    // MPI_Win win;
    bool m_device = false;
    std::size_t aligned_size = 0;
    mi_arena_id_t arena_id{};
    mi_heap_t* heap = nullptr;
    mi_stats_t stats;
};
