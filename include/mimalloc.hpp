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
* The free(void* ...) is overloaded by free(Mimalloc ...) to make sure the memory is unpinned before freeing it.
* Set "prefix" to the namespace of the allocator.
*/
#define ALLOC(allocator, deallocator) \
void* ext_##allocator##_aligned(std::size_t size){ \
    if ( #allocator == "std_malloc" ) { return aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size); } \
    void* tmp = allocator(size); \
    fmt::print("{} : Raw ptr. \n", tmp); \
    void* aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size/2, tmp, size); \
    fmt::print("{} : Aligned ptr. \n", aligned_ptr); \
    return aligned_ptr; \
} \
void ext_##deallocator(Mimalloc mim){ \
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


class Mimalloc {
public:
  /**
   * @brief Manages a particular memory arena. 
   * Set numa_node to 0 if single numa node, ignore if unknown.
   */
    Mimalloc(void* addr, const std::size_t size, const bool is_committed = false,
            const bool is_zero = true, int numa_node = -1);

    // Leave it undeleted to keep allocated blocks
    ~Mimalloc() {}

    std::size_t AlignedSize() const { return aligned_size; }

    void* AlignedAddress() const { return aligned_address; }

    void* allocate(const std::size_t bytes, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

#if PMIMALLOC_ENABLE_DEVICE

    void* device_allocate(std::size_t size);

    // void* device_reallocate(void* ptr, std::size_t size);

    void device_deallocate(void* ptr) noexcept;

#endif

    /**
    * @brief Set bool pin to true to pin the memory, false to unpin it.
    */ 
    int pin_or_unpin(bool pin);

    void pin(){ int success = pin_or_unpin(true); }

    void unpin(){ int success = pin_or_unpin(false); }

    std::size_t getAllocatedSize(void* ptr) { return mi_usable_size(ptr); }

private:
    void* aligned_address = nullptr;
    // int key;
    // MPI_Win win;
    std::size_t aligned_size = 0;
    mi_arena_id_t arena_id{};
    mi_heap_t* heap = nullptr;
    mi_stats_t stats;
};
