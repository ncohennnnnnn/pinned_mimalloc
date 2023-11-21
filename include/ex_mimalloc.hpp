#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <utility>

#include <mimalloc.h>
#include <mimalloc/internal.h>
// #include <mimalloc/atomic.h>
// #include <../src/bitmap.h>

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


static inline void* get_prim_tls_slot(size_t slot) noexcept;
mi_threadid_t get_thread_id(void) noexcept;

const std::size_t nb_threads = 3;

class ex_mimalloc {
public:
    ex_mimalloc() noexcept = default;

    ex_mimalloc(void* ptr, const std::size_t size, const int numa_node);

    template<typename Context>
    ex_mimalloc( const Context& C );

    ex_mimalloc(const ex_mimalloc& m) = delete;

    ~ex_mimalloc() {}
    
    void* allocate(const std::size_t size, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void  deallocate(void* ptr, std::size_t size = 0);

    std::size_t get_usable_size(void* ptr) { return mi_usable_size(ptr); }

private:
    // std::pair<int,int> m_threads[nb_threads];
    std::pair<int,std::thread::id> m_threads[nb_threads];
    mi_arena_id_t m_arena_id{};
    mi_heap_t* m_heap[nb_threads];
    mi_stats_t m_stats;
};