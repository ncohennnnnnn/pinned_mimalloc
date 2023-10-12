#include <pmimalloc.hpp>
#include <selector.hpp>

#include <mpi.h>

int main() {
    // MPI_Init(NULL, NULL);

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint32_t heap_sz = 1 << 26;

    // void* std_heap_start = ext_std_alloc_aligned(heap_sz); // maybe also call the Mimalloc constructor
    // void* MPI_heap_start = ext_MPI_alloc_aligned(heap_sz); // maybe also call the Mimalloc constructor
    void* hip_heap_start = ext_hip_alloc_aligned(heap_sz);  // maybe also call the Mimalloc constructor

    numa_available();
    // int std_numa_node = get_node(std_heap_start);
    // int MPI_numa_node = get_node(MPI_heap_start);
    int hip_numa_node = get_node(hip_heap_start);


    // pmimalloc std_ext_alloc(std_heap_start, heap_sz, true, false, std_numa_node);
    // pmimalloc MPI_ext_alloc(MPI_heap_start, heap_sz, true, false, MPI_numa_node);
    pmimalloc hip_ext_alloc(hip_heap_start, heap_sz, true, false, hip_numa_node, true);

    // void* std_p1 = std_ext_alloc.allocate(32);
    // void* MPI_p1 = MPI_ext_alloc.allocate(32);
    void* hip_p1 = hip_ext_alloc.allocate(32);
    // void* std_p2 = std_ext_alloc.allocate(48);
    // void* MPI_p2 = MPI_ext_alloc.allocate(48);
    void* hip_p2 = hip_ext_alloc.allocate(48);
    // std_ext_alloc.deallocate(std_p1);
    // MPI_ext_alloc.deallocate(MPI_p1);
    hip_ext_alloc.deallocate(hip_p1);
    // std_ext_alloc.deallocate(std_p2);
    // MPI_ext_alloc.deallocate(MPI_p2);
    hip_ext_alloc.deallocate(hip_p2);

    // ext_std_free(std_ext_alloc);
    // ext_MPI_free(MPI_ext_alloc);
    ext_hip_free(hip_ext_alloc);

    mi_collect(true);
    mi_stats_print(NULL);

    // MPI_Finalize();

    return 0;
}
