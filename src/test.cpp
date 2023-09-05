#include <config.hpp>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <mimalloc.h>
#include <mimalloc.hpp>

int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((size_t)ptr & ~((size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("move page failed.\n");
        return -1;
    }
    return numa_node[0];
}

int main() {
    numa_available();

    uint32_t heap_sz = ext_heap_sz;
    void* heap_start = std::malloc(heap_sz);

    int numa_node = get_node(heap_start);

    Mimalloc ext_alloc(heap_start, heap_sz, false, true, numa_node);

    void* p1 = ext_alloc.allocate(32);
    void* P1 = ext_alloc.allockate(32);
    void* p2 = ext_alloc.allocate(48);
    void* P2 = ext_alloc.allockate(48);
    ext_alloc.deallocate(p1);
    ext_alloc.deallockate(P1);
    ext_alloc.deallocate(p2);
    ext_alloc.deallockate(P2);
    std::free(heap_start);

    // ext_alloc.~Mimalloc();

    mi_collect(true);
    mi_stats_print(NULL);

    return 0;
}
