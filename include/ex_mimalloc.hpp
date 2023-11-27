#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include <mimalloc.h>
#include <mimalloc/internal.h>

class ex_mimalloc
{
public:
    ex_mimalloc() noexcept = default;

    ex_mimalloc(void* ptr, const std::size_t size, const int numa_node);

    template <typename Context>
    ex_mimalloc(const Context& C);

    ex_mimalloc(const ex_mimalloc& m) = delete;

    ~ex_mimalloc();

    void* allocate(const std::size_t size, const std::size_t alignment = 0);

    void* reallocate(void* ptr, std::size_t size);

    void deallocate(void* ptr, std::size_t size = 0);

    // std::size_t get_usable_size();

private:
    mi_arena_id_t m_arena_id{};
    mi_stats_t m_stats;
};
