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

template<typename C>
pmimalloc::pmimalloc(const std::size_t size){
    if (size == 0) { return *this; }

    pmimalloc::m_address = C::allocate_aligned(size);
    pmimalloc::m_size    = size;

    /* Pin the allocated memory, if not already by device runtime. */
    C::pin(m_address, m_size);

    /* Find NUMA node if not known before */
    if ( numa_node == -1 ) { pmimalloc::m_numa_node = get_node(m_address); }

    /** @brief Create the mimalloc arena
    *  @param exclusive allows allocations if specifically for this arena
    * TODO : @param is_large could be an option
    */
    bool success = mi_manage_os_memory_ex(m_address, m_size, true, false, false, m_numa_node, true, &m_arena_id);
    if (!success) {
        fmt::print("{} : [error] pmimalloc failed to create the arena. \n", m_address);
        m_address = nullptr;
    }

    /* Associate a heap to the arena */
    m_heap = mi_heap_new_in_arena(m_arena_id);
    if (m_heap == nullptr) {
        fmt::print("{} : [error] pmimalloc failed to create the heap. \n", m_address);
        m_address = nullptr;
    }

    /* Do not use OS memory for allocation (but only pre-allocated arena). */
    mi_option_set(mi_option_limit_os_alloc, 1);

    /* Window creation or registering from the context. */
    C::register(m_address, m_size);
}

template<typename C>
pmimalloc::pmimalloc(const pmimalloc<C>& other, const std::size_t size){
    return pmimalloc<C>(size);
}

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

// void pmimalloc::win_free() {
// #if PMIMALLOC_WITH_MPI
//     int res_win = MPI_Win_free(&m_win);
//     if (res_win != MPI_SUCCESS) {
//         char err_buffer[MPI_MAX_ERROR_STRING];
//         int resultlen;
//         MPI_Error_string(res_win, err_buffer,&resultlen);
//         fmt::print(stderr,err_buffer);
//         MPI_Finalize();
//     } else { fmt::print("{} : MPI Window freed \n", m_address); }
// #endif
// }

//     /* MPI window creation. */
//     int res_win = MPI_Win_create(&m_address, m_size, 1 , MPI_INFO_NULL, MPI_COMM_WORLD, &m_win);
//     if (res_win != MPI_SUCCESS) {
//         char err_buffer[MPI_MAX_ERROR_STRING];
//         int resultlen;
//         MPI_Error_string(res_win, err_buffer,&resultlen);
//         fmt::print(stderr,err_buffer);
//         MPI_Finalize();
//     } else { fmt::print("{} : MPI Window created \n", m_address); }
// }
