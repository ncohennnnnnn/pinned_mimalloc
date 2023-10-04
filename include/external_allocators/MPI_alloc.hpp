#include <mimalloc.hpp>
#include <mpi.h>

void* MPI_alloc(size_t size) { 
    void* rtn = nullptr;
    MPI_Alloc_mem(size, MPI_INFO_NULL, rtn);
    return rtn;
}
void MPI_free(void* ptr) { MPI_Free_mem(ptr); }

ALLOC(MPI_alloc, MPI_free)