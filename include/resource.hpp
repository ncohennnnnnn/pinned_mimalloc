#include <context.hpp>

#if WITH_MIMALLOC

#include <cstdint>
// #include <cstdlib>
#include <stdexcept>
// #include <sys/mman.h>
// #include <cstring>
#include <iostream>
#include <unistd.h>
// #include <errno.h>

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

#endif


template<typename context>
class resource{
public:
    using key_t = context::key_t;

    resource(std::size_t size, bool pin, std::size_t alignement = 0);

    std::size_t get_usable_size(void* ptr);

    void* get_address() { return m_context.get_address(); }

    std::size_t get_size() { return m_context.get_size(); }

    template<typename T>
    key_t get_key(T* ptr) { return m_context.get_key(ptr); }

    void* allocate(const std::size_t size, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

private:
    context m_context;
#if WITH_MIMALLOC
    mi_arena_id_t m_arena_id{};
    mi_heap_t* m_heap = nullptr;
    mi_stats_t m_stats;
#endif
};
