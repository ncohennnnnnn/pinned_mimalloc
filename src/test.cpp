#include <mimalloc.hpp>
#include <mpi.h>

#if ALLOCATE_WITH_MPI
#include "external_allocators/MPI_alloc.hpp"
#endif

#if ALLOCATE_WITH_STD
#include "external_allocators/std_alloc.hpp"
#endif

int main() {
    MPI_Init(NULL, NULL);

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint32_t heap_sz = 1 << 27;

    void* std_heap_start = ext_std_alloc_aligned(heap_sz); // maybe also call the Mimalloc constructor

    numa_available();
    int std_numa_node = get_node(std_heap_start);


    Mimalloc std_ext_alloc(std_heap_start, heap_sz, true, false, std_numa_node);
    std::cout << "Here 1" << std::endl;
    void* MPI_heap_start = ext_MPI_alloc_aligned(heap_sz); // maybe also call the Mimalloc constructor
    std::cout << "Here 2" << std::endl;
    int MPI_numa_node = get_node(MPI_heap_start);
    Mimalloc MPI_ext_alloc(MPI_heap_start, heap_sz, true, false, MPI_numa_node);

    void* std_p1 = std_ext_alloc.allocate(32);
    // void* MPI_p1 = MPI_ext_alloc.allocate(32);
    void* std_p2 = std_ext_alloc.allocate(48);
    // void* MPI_p2 = MPI_ext_alloc.allocate(48);
    std_ext_alloc.deallocate(std_p1);
    // MPI_ext_alloc.deallocate(MPI_p1);
    std_ext_alloc.deallocate(std_p2);
    // MPI_ext_alloc.deallocate(MPI_p2);

    ext_std_free(std_ext_alloc);
    ext_MPI_free(MPI_ext_alloc);

    mi_collect(true);
    mi_stats_print(NULL);

    MPI_Finalize();

    return 0;
}
