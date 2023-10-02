#include <mimalloc.hpp>


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

Mimalloc::Mimalloc(void* addr, const std::size_t size , const bool is_committed,
            const bool is_zero, int numa_node){
    // Doesn't consist of large OS pages
    bool is_large = false;

    Mimalloc::aligned_size = size ;
    Mimalloc::aligned_address = addr;

    // Find NUMA node if not known before 
    if ( numa_node == -1 ) { numa_node = get_node(aligned_address); }

    // Create the mimalloc arena
    bool success = mi_manage_os_memory_ex(aligned_address, aligned_size, is_committed,
                                        is_large, is_zero, numa_node, true, &arena_id);
    if (!success) { // TODO : add error throw
        fmt::print("{} : [error] Mimalloc failed to create the arena. \n", aligned_address);
        Mimalloc::aligned_address = nullptr;
    }

    // Associate a heap to the arena
    heap = mi_heap_new_in_arena(arena_id);
    if (heap == nullptr) { // TODO : add error throw
        fmt::print("{} : [error] Mimalloc failed to create the heap. \n", aligned_address);
        Mimalloc::aligned_address = nullptr;
    }

    // Do not use OS memory for allocation (but only pre-allocated arena)
    mi_option_set(mi_option_limit_os_alloc, 1);

    // Pin the allocated memory
    Mimalloc::pin();
}

/*
TODO : discriminate between device and host alloc, specialize on host alloc and maybe let
some other allocator or context do the device allocation job.
*/
void* Mimalloc::allocate(const std::size_t bytes, const std::size_t alignment) {
    void* rtn = nullptr;
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(heap, bytes, alignment);
    } else {
        rtn = mi_heap_malloc(heap, bytes);
    }
    fmt::print("{} : Memory allocated. \n", rtn);
    return rtn;
}

void* Mimalloc::reallocate(void* ptr, std::size_t size ) {
    return mi_heap_realloc(heap, ptr, size );
}

void Mimalloc::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
}

int Mimalloc::pin_or_unpin(bool pin){  // TODO : add error throw
    int success;
    std::string str;
    if ( pin ) { 
        success = mlock(aligned_address, aligned_size); // CHANGE MLOCK TO A MORE GENERAL LOCK
        str = "pin";
    }
    else { 
        success = munlock(aligned_address, aligned_size);  // CHANGE MUNLOCK TO A MORE GENERAL LOCK
        str = "unpin";
    }
    if ( success != 0) { 
        fmt::print("{} : [error] Mimalloc failed to {} the allocated memory :  ", aligned_address, str);
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
