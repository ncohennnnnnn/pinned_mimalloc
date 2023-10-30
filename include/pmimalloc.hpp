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
// #define ALLOC(allocator, deallocator) \
// void* ext_##allocator##_aligned(std::size_t size){ \
//     void* aligned_ptr; \
//     if ( #allocator == "std_alloc") { aligned_ptr = aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size); } \
//     else { \
//         std::size_t size_buff = 2*size; \
//         void* tmp = allocator(size_buff); \
//         aligned_ptr = std::align(MIMALLOC_SEGMENT_ALIGNED_SIZE, size, tmp, size_buff); \
//     } \
//     fmt::print("{} : Aligned {}ated ptr. \n", aligned_ptr, #allocator); \
//     return aligned_ptr; \
// } \
// void ext_##deallocator(pmimalloc mim){ \
//     mim.unpin(); \
//     mim.win_free(); \
//     deallocator(mim.get_Address()); \
// } \

int get_node(void* ptr);

/** @brief Creates a mimalloc arena from externally allocated memory.
* TODO: throw in some concepts.
*/
template<typename context>
class pmimalloc {

public:
    using this_type = pmimalloc<context>;

    // using propagate_on_container_copy_assignment = std::true_type;
    // using propagate_on_container_move_assignment = std::true_type;
    // using propagate_on_container_swap = std::true_type;
    // using is_always_equal = std::false_type;

    template<typename C>
    struct other_pmimalloc {
        using other = pmimalloc<C>;
    };

    template<typename C>
    using rebind = other_pmimalloc<C>;

public:
    pmimalloc(const context context, const std::size_t size = 0);

    template<typename C>
    pmimalloc(const pmimalloc<C>& other, const std::size_t size = 0);

    /* Leave it undeleted to keep allocated blocks */
    ~pmimalloc() {}

    friend bool operator==(const pmimalloc& lhs, const pmimalloc& rhs) noexcept
    {
        return (lhs.m_context == rhs.m_context);
    }

    friend bool operator!=(const pmimalloc& lhs, const pmimalloc& rhs) noexcept
    {
        return (lhs.m_context != rhs.m_context);
    }

    friend void swap(allocator& lhs, allocator& rhs) noexcept
    {
        using std::swap;
        swap(lhs.m_heap, rhs.m_heap);
        swap(lhs.m_numa_node, rhs.m_numa_node);
    }

    void* allocate(const std::size_t bytes, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

    std::size_t get_UsableSize(void* ptr) { return mi_usable_size(ptr); }

    std::size_t get_TotalSize() const { return m_size; }

    void* get_Address() const { return m_address; }

    auto get_Key() { return m_key; };

private:
    void* m_address = nullptr;
    std:size_t m_numa_node = 0;
    context* m_context;
    auto m_key;
    std::size_t m_size = 0;
    mi_arena_id_t m_arena_id{};
    mi_heap_t* m_heap = nullptr;
    mi_stats_t m_stats;
};