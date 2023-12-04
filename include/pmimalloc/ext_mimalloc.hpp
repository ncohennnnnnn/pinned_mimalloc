#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include <fmt/core.h>

#include <mimalloc.h>
/* TODO: Put this under a debug option */
#include <mimalloc/internal.h>

#if USE_UNORDERED_MAP
thread_local std::unordered_map<mi_arena_id_t, mi_heap_t*> tl_ext_mimalloc_heaps;
#endif
#if USE_TL_VECTOR
# include <pmimalloc/indexed_tl_ptr.hpp>
// auto maker = []() {
//     // mi_heap_t* heap = nullptr;
//     // return *heap;
// };
// auto deleter = [](/*mi_heap_t* ptr*/) {};
#endif

class ext_mimalloc
{
public:
    ext_mimalloc() {}

    ext_mimalloc(void* ptr, const std::size_t size, const int numa_node);

    template <typename Context>
    ext_mimalloc(const Context& C);

    ext_mimalloc(const ext_mimalloc& m) = delete;

    ~ext_mimalloc();

    void* allocate(const std::size_t size, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

    mi_arena_id_t get_arena();

    mi_heap_t* get_heap();

    void set_heap();

    bool heap_exists();

    // bool is_in_arena();

private:
    mi_arena_id_t m_arena_id{};
    mi_stats_t m_stats;
#if USE_TL_VECTOR
    indexed_tl_ptr<mi_heap_t> m_heaps;
#endif
};
