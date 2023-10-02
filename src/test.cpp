#include <config.hpp>

#include <mimalloc.h>
#include <mimalloc.hpp>

#define std_malloc(x)           std::malloc(x)
#define std_free(x)             std::free(x)

ALLOC(std_malloc, std_free);

int main() {
    uint32_t heap_sz = ext_heap_sz;
    void* heap_start = std_malloc_aligned(heap_sz);

    numa_available();
    int numa_node = get_node(heap_start);

    Mimalloc ext_alloc(heap_start, heap_sz, true, false, numa_node);

    void* p1 = ext_alloc.allocate(32);
    void* p2 = ext_alloc.allocate(48);
    ext_alloc.deallocate(p1);
    ext_alloc.deallocate(p2);

    free(ext_alloc);

    mi_collect(true);
    mi_stats_print(NULL);

    return 0;
}
