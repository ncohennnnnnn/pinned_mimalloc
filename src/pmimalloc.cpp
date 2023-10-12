#include <pmimalloc.hpp>

int get_node(void* ptr){
    int numa_node[1] = {-1};
    void* page = (void*)((std::size_t)ptr & ~((std::size_t)getpagesize()-1));
    int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
    if (err == -1) {
        fmt::print("Move page failed. \n");
        return -1;
    }
    return numa_node[0];
}

pmimalloc::pmimalloc(void* addr, const std::size_t size, const bool is_committed,
            const bool is_zero, int numa_node, const bool device){
    pmimalloc::device = device;
    
    /* Doesn't consist of large OS pages. */
    bool is_large = false;

   
    pmimalloc::aligned_size = size ;
    pmimalloc::aligned_address = addr;

    /* Pin the allocated memory, if not already by device runtime. */
    if( device == true) { pmimalloc::device = device; }
    else { pmimalloc::pin(); }

    // Find NUMA node if not known before 
    if ( numa_node == -1 ) { numa_node = get_node(aligned_address); }

    // Create the mimalloc arena
    bool success = mi_manage_os_memory_ex(aligned_address, aligned_size, is_committed,
                                        is_large, is_zero, numa_node, true, &arena_id);
    if (!success) { // TODO : add error throw
        fmt::print("{} : [error] pmimalloc failed to create the arena. \n", aligned_address);
        pmimalloc::aligned_address = nullptr;
    }

    // Associate a heap to the arena
    heap = mi_heap_new_in_arena(arena_id);
    if (heap == nullptr) { // TODO : add error throw
        fmt::print("{} : [error] pmimalloc failed to create the heap. \n", aligned_address);
        pmimalloc::aligned_address = nullptr;
    }

    // Do not use OS memory for allocation (but only pre-allocated arena)
    mi_option_set(mi_option_limit_os_alloc, 1);

    // // For now we only use MPI to get the RMA key
    // MPI_Win_lock_all(0, win);
    // key = MPI_Win_shared_query(win, &rank, size, 1);
    // MPI_Win_unlock_all(win);

    // TODO : Choose the way to pin, i.e. depends on the network if pinned
    // Pin the allocated memory
    // pmimalloc::pin();
}

/* TODO : discriminate between contexts ? */
void* pmimalloc::allocate(const std::size_t bytes, const std::size_t alignment) {
    void* rtn = nullptr;
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(heap, bytes, alignment);
    } else {
        rtn = mi_heap_malloc(heap, bytes);
    }
    fmt::print("{} : Memory allocated. \n", rtn);
    return rtn;
}

void* pmimalloc::reallocate(void* ptr, std::size_t size ) {
    return mi_heap_realloc(heap, ptr, size );
}

void pmimalloc::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
}

#if !PMIMALLOC_ENABLE_DEVICE

void* device_allocate(std::size_t size) {}

// void* device_reallocate(void* ptr, std::size_t size) {}

void device_deallocate(void* ptr) noexcept {}

int pmimalloc::pin_or_unpin(bool pin){  // TODO : add error throw
    int success;
    std::string str;
    if ( pin ) { 
        success = mlock(aligned_address, aligned_size); // TODO : Adapt the pinning to the backend
        str = "pin";
    }
    else { 
        success = munlock(aligned_address, aligned_size);  // TODO : Adapt the unpinning to the backend
        str = "unpin";
    }
    if ( success != 0) { 
        fmt::print("{} : [error] pmimalloc failed to {} the allocated memory :  ", aligned_address, str);
        if        (errno == EAGAIN) {
            fmt::print("EAGAIN. \n (Some or all of the specified address range could not be locked.) \n");
        } else if (errno == EINVAL) {
            fmt::print("EINVAL. \n (The result of the addition addr+len was less than addr. addr = {} and len = {})\n", aligned_address, aligned_size);
        } else if (errno == ENOMEM) {
            fmt::print("ENOMEM. \n (Some of the specified address range does not correspond to mapped pages in the address space of the process.) \n");
        } else if (errno == EPERM ) {
            fmt::print("EPERM. \n (The caller was not privileged.) \n");
        }
    } 
    else { fmt::print("{} : Memory {}ned. \n", aligned_address, str); } 

    return success;
}

#else

int pmimalloc::pin_or_unpin(bool pin) { return 0; }

#endif
