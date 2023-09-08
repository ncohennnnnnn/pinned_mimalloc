#include <config.hpp>
#include <stdio.h>
#include <assert.h>

#include <fmt/core.h>

#include <numa.h>
#include <mimalloc.h>
#include <mimalloc.hpp>

#define std_malloc(x)           std::malloc(x)
#define std_free(x)             std::free(x)

ALIGNED(std_malloc, std_free);

int main() {
    uint32_t heap_sz = ext_heap_sz;
    void* heap_start = std_malloc_aligned(heap_sz);

    numa_available();
    int numa_node = get_node(heap_start);

    Mimalloc ext_alloc(heap_start, heap_sz, false, true, numa_node);

    void* p1 = ext_alloc.allocate(32);
    void* p2 = ext_alloc.allocate(48);
    ext_alloc.deallocate(p1);
    ext_alloc.deallocate(p2);
    std::free(heap_start);

    // void* heap_start_2 = std_malloc_aligned(heap_sz);
    // bool pin = mlock(heap_start_2, heap_sz);
    // fmt::print( "mlock successful : {} \n", pin );
    // bool unpin = munlock(heap_start_2, heap_sz);
    // fmt::print( "munlock successful : {} \n", pin );
    // std::free(heap_start_2);

    mi_collect(true);
    mi_stats_print(NULL);

    return 0;
}
