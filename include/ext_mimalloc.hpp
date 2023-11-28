#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include <mimalloc.h>
/* TODO: get rid of this include */
#include <mimalloc/internal.h>

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

    mi_heap_t* get_heap();

    // std::size_t get_usable_size();

private:
    mi_arena_id_t m_arena_id{};
    mi_stats_t m_stats;
};
