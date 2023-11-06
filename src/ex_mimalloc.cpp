#include <ex_mimalloc.hpp>
#include <fmt/core.h>


ex_mimalloc::ex_mimalloc(void* ptr, const std::size_t size, const int numa_node){
    if (size != 0) {
        /** @brief Create the ex_mimalloc arena
        * @param exclusive allows allocations if specifically for this arena
        * 
        * TODO: @param is_large could be an option
        */
        bool success = mi_manage_os_memory_ex(ptr, size, true, false, false, numa_node, true, &m_arena_id);
        if (!success) {
            fmt::print("{} : [error] ex_mimalloc failed to create the arena. \n", ptr);
        }

        /* Associate a heap to the arena */
        m_heap = mi_heap_new_in_arena(m_arena_id);
        if (m_heap == nullptr) {
            fmt::print("{} : [error] ex_mimalloc failed to create the heap. \n", ptr);
        }

        /* Do not use OS memory for allocation (but only pre-allocated arena). */
        mi_option_set(mi_option_limit_os_alloc, 1);
    }
}

template<typename Context>
ex_mimalloc::ex_mimalloc( const Context& C )
{
    ex_mimalloc{C.get_address(), C.get_size(), C.get_numa_node()};
}


void* ex_mimalloc::allocate(const std::size_t size, const std::size_t alignment) {
    void* rtn = nullptr;
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(m_heap, size, alignment);
    } else {
        rtn = mi_heap_malloc(m_heap, size);
    }
    fmt::print("{} : Memory allocated. \n", rtn);
    return rtn;
}

void* ex_mimalloc::reallocate(void* ptr, std::size_t size ) {
    return mi_heap_realloc(m_heap, ptr, size );
}

void ex_mimalloc::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
    fmt::print("{} : Memory deallocated. \n", ptr);
}

// void ex_mimalloc::win_free() {
// #if ex_mimalloc_WITH_MPI
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
