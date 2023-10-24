#include <pmimalloc.hpp>

#if PMIMALLOC_WITH_CUDA
#include <cuda_runtime.h>
#endif

int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((std::size_t)ptr & ~((std::size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("Move page failed from get_node(). \n");
        return -1;
    }
    return numa_node[0];
}

pmimalloc::pmimalloc(void* addr, const std::size_t size, const bool device_allocated, const int numa_node){
    pmimalloc::m_device_allocated = device_allocated;
    pmimalloc::m_size = size;
    pmimalloc::m_address = addr;

    /* Pin the allocated memory, if not already by device runtime. */
    pmimalloc::pin();

    /* Find NUMA node if not known before */
    if ( numa_node == -1 ) { pmimalloc::m_numa_node = get_node(m_address); }

    /** @brief Create the mimalloc arena
    @param exclusive allows allocations if specifically for this arena
    TODO : @param is_large could be an option
    */
    bool success = mi_manage_os_memory_ex(m_address, m_size, true, false, false, m_numa_node, true, &m_arena_id);
    if (!success) { /* TODO : add error throw */
        fmt::print("{} : [error] pmimalloc failed to create the arena. \n", m_address);
        m_address = nullptr;
    }

    /* Associate a heap to the arena */
    m_heap = mi_heap_new_in_arena(m_arena_id);
    if (m_heap == nullptr) { // TODO : add error throw
        fmt::print("{} : [error] pmimalloc failed to create the heap. \n", m_address);
        m_address = nullptr;
    }

    /* Do not use OS memory for allocation (but only pre-allocated arena) */
    mi_option_set(mi_option_limit_os_alloc, 1);

    /* For now we only use MPI to create windows
    * TODO : Add the possibility to add a different communicator
    */
    int res_win = MPI_Win_create(&m_address, m_size, 1 , MPI_INFO_NULL, MPI_COMM_WORLD, &m_win);
    if (res_win != MPI_SUCCESS) {
        char err_buffer[MPI_MAX_ERROR_STRING];
        int resultlen;
        MPI_Error_string(res_win, err_buffer,&resultlen);
        fmt::print(stderr,err_buffer);
        MPI_Finalize();
    } else { fmt::print("{} : MPI Window created \n", m_address); }
}

/* TODO : discriminate between contexts ? */
void* pmimalloc::allocate(const std::size_t bytes, const std::size_t alignment) {
    void* rtn = nullptr;
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(m_heap, bytes, alignment);
    } else {
        rtn = mi_heap_malloc(m_heap, bytes);
    }
    fmt::print("{} : Memory allocated. \n", rtn);
    return rtn;
}

void* pmimalloc::reallocate(void* ptr, std::size_t size ) {
    return mi_heap_realloc(m_heap, ptr, size );
}

void pmimalloc::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
    fmt::print("{} : Memory deallocated. \n", ptr);
}

void _std_pin_or_unpin(bool pin, void* ptr, const size_t size){ /* TODO : add error throw */
    int success;
    std::string str;
    if (pin) {
        success = mlock(ptr, size); /* TODO : Adapt the pinning to the backend */
        str = "pin";
    } else { 
        success = munlock(ptr, size);  /* TODO : Adapt the unpinning to the backend */
        str = "unpin";
    }
    if (success != 0) { 
        fmt::print("{} : [error] pmimalloc failed to {} the allocated memory :  ", ptr, str);
        if        (errno == EAGAIN) {
            fmt::print("EAGAIN. \n (Some or all of the specified address range could not be locked.) \n");
        } else if (errno == EINVAL) {
            fmt::print("EINVAL. \n (The result of the addition addr+len was less than addr. addr = {} and len = {})\n", ptr, size);
        } else if (errno == ENOMEM) {
            fmt::print("ENOMEM. \n (Some of the specified address range does not correspond to mapped pages in the address space of the process.) \n");
        } else if (errno == EPERM ) {
            fmt::print("EPERM. \n (The caller was not privileged.) \n");
        }
    } else { fmt::print("{} : Memory {}ned. \n", ptr, str); } 
}

#if PMIMALLOC_WITH_CUDA
void _cuda_pin_or_unpin(bool pin, void* ptr, const size_t size){ /* TODO : add error throw */
    cudaError_t cudaStatus;
    if (pin) {
        /* TODO: Choose the appropriate flags */
        cudaStatus = cudaHostRegister( ptr, size, cudaHostRegisterDefault ) ;
        if (cudaStatus != cudaSuccess) {
            fmt::print("cudaHostRegister failed: {} \n", cudaGetErrorString(cudaStatus));
        } else { fmt::print("{} : Memory pinned (by CUDA). \n", ptr); }
    } else {
        /* TODO: Choose the appropriate flags */
        cudaStatus = cudaHostUnregister( ptr ) ;
        if (cudaStatus != cudaSuccess) {
            fmt::print("cudaHostUnregister failed: {} \n", cudaGetErrorString(cudaStatus));
        } else { fmt::print("{} : Memory unpinned (by CUDA). \n", ptr); }
    }
}
#endif

void pmimalloc::pin_or_unpin(bool pin){
    if( !m_device_allocated ) {
#if PMIMALLOC_WITH_CUDA
        _cuda_pin_or_unpin(pin, m_address, m_size);
#else
        _std_pin_or_unpin(pin, m_address, m_size); 
#endif
    }
}

void pmimalloc::pin(){ pin_or_unpin(true); }

void pmimalloc::unpin(){ pin_or_unpin(false); }

void pmimalloc::win_free() {
#if PMIMALLOC_WITH_MPI
    int res_win = MPI_Win_free(&m_win);
    if (res_win != MPI_SUCCESS) {
        char err_buffer[MPI_MAX_ERROR_STRING];
        int resultlen;
        MPI_Error_string(res_win, err_buffer,&resultlen);
        fmt::print(stderr,err_buffer);
        MPI_Finalize();
    } else { fmt::print("{} : MPI Window freed \n", m_address); }
#endif
}