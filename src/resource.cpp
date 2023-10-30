#include <resource.hpp>

resource::resource(std::size_t size, bool pin, std::size_t alignement){
#if WITH_MIMALLOC
    if (size == 0) { return *this; }

    m_context{size, pin};
    
    address = m_context.get_address();

    /** @brief Create the mimalloc arena
    *  @param exclusive allows allocations if specifically for this arena
    *  @todo: @param is_large could be an option
    */
    bool success = mi_manage_os_memory_ex(address, size, true, false, false, m_context.get_numa_node(), true, &m_arena_id);
    if (!success) {
        fmt::print("{} : [error] pmimalloc failed to create the arena. \n", address);
    }

    /* Associate a heap to the arena */
    m_heap = mi_heap_new_in_arena(m_arena_id);
    if (m_heap == nullptr) {
        fmt::print("{} : [error] pmimalloc failed to create the heap. \n", address);
    }

    /* Do not use OS memory for allocation (but only pre-allocated arena). */
    mi_option_set(mi_option_limit_os_alloc, 1);
#else
    m_context{size, pin, alignement};
    return *this;
#endif
}

std::size_t resource::get_usable_size(void* ptr) { 
#if WITH_MIMALLOC
    return mi_usable_size(ptr); 
#else
    return sizeof(ptr);
#endif
}

void* resource::allocate(const std::size_t bytes, const std::size_t alignment) {
    void* rtn = nullptr;
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(m_heap, bytes, alignment);
    } else {
        rtn = mi_heap_malloc(m_heap, bytes);
    }
    fmt::print("{} : Memory allocated. \n", rtn);
    return rtn;
}

void* resource::reallocate(void* ptr, std::size_t size ) {
    return mi_heap_realloc(m_heap, ptr, size );
}

void resource::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
    fmt::print("{} : Memory deallocated. \n", ptr);
}
