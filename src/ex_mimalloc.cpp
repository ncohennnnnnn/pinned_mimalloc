#include <ex_mimalloc.hpp>
#include <task_group.hpp>

#include <fmt/core.h>

#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__IBMC__) || defined(__INTEL_COMPILER) ||    \
    defined(__clang__)
# ifndef unlikely
#  define unlikely(x_) __builtin_expect(!!(x_), 0)
# endif
# ifndef likely
#  define likely(x_) __builtin_expect(!!(x_), 1)
# endif
#else
# ifndef unlikely
#  define unlikely(x_) (x_)
# endif
# ifndef likely
#  define likely(x_) (x_)
# endif
#endif

// void mi_heap_destroy(mi_heap_t* heap);
// using unique_tls_heap = std::unique_ptr<mi_heap_t, void (*)(mi_heap_t *)>;
thread_local mi_heap_t* thread_local_ex_mimalloc_heap{nullptr};

ex_mimalloc::ex_mimalloc(void* ptr, const std::size_t size, const int numa_node)
{
    if (size != 0)
    {
        /** @brief Create the ex_mimalloc arena
     * @param exclusive allows allocations if specifically for this arena
     * @param is_committed set to true
     * @param is_large could be an option
     */
        bool success =
            mi_manage_os_memory_ex(ptr, size, true, false, true, numa_node, true, &m_arena_id);
        if (!success)
        {
            fmt::print("{} : [error] ex_mimalloc failed to create the arena. \n", ptr);
        }
        else
        {
            fmt::print("{} : Mimalloc arena created \n", ptr);
        }
        /* Do not use OS memory for allocation (but only pre-allocated arena). */
        mi_option_set(mi_option_limit_os_alloc, 1);

        /* OS tag to assign to mimalloc'd memory. */
        mi_option_enable(mi_option_os_tag);

        if (!thread_local_ex_mimalloc_heap)
        {
            // auto my_delete = [](mi_heap_t* heap) {
            //     fmt::print("ex_mimalloc:: NOT Deleting heap (it's safe) {}\n", (void*) (heap));
            //     mi_heap_destroy(heap);
            // };
            thread_local_ex_mimalloc_heap = mi_heap_new_in_arena(m_arena_id);
            // unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
            fmt::print("ex_mimalloc:: New thread local backing heap {} ",
                (void*) (thread_local_ex_mimalloc_heap));
        }
        mi_heap_set_default(thread_local_ex_mimalloc_heap);
    }
}

template <typename Context>
ex_mimalloc::ex_mimalloc(const Context& C)
{
    ex_mimalloc{C.get_address(), C.get_size(), C.get_numa_node()};
}

void* ex_mimalloc::allocate(const std::size_t size, const std::size_t alignment)
{
    if (!thread_local_ex_mimalloc_heap)
    {
        // auto my_delete = [](mi_heap_t* heap) {
        //     fmt::print("ex_mimalloc:: NOT Deleting heap (it's safe) {}\n", (void*) (heap));
        //     mi_heap_destroy(heap);
        // };
        thread_local_ex_mimalloc_heap = mi_heap_new_in_arena(m_arena_id);
        // unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
        fmt::print(
            "ex_mimalloc:: New thread local heap {} ", (void*) (thread_local_ex_mimalloc_heap));
    }

    void* rtn = nullptr;
    if (unlikely(alignment))
    {
        rtn = mi_heap_malloc_aligned(thread_local_ex_mimalloc_heap, size, alignment);
    }
    else
    {
        rtn = mi_heap_malloc(thread_local_ex_mimalloc_heap, size);
    }
    //    fmt::print("{} : Memory allocated with size {} from heap {} \n", rtn, size,
    //        (void*) (thread_local_ex_mimalloc_heap));
    return rtn;
}

void* ex_mimalloc::reallocate(void* ptr, std::size_t size)
{
    if (!thread_local_ex_mimalloc_heap)
    {
        std::cout << "ERROR!!! how can this happen" << std::endl;
    }
    return mi_heap_realloc(thread_local_ex_mimalloc_heap, ptr, size);
}

void ex_mimalloc::deallocate(void* ptr, std::size_t size)
{
    if (likely(ptr))
    {
        if (unlikely(size))
        {
            mi_free_size(ptr, size);
        }
        else
        {
            mi_free(ptr);
        }
    }
    //    fmt::print("{} : Memory deallocated. \n", ptr);
}

ex_mimalloc::~ex_mimalloc()
{
    if (!thread_local_ex_mimalloc_heap)
    {
        std::cout << "ERROR!!! how can this happen" << std::endl;
    }
    else
    {
        if (thread_local_ex_mimalloc_heap->page_count != 0)
        {
            fmt::print("Heap not empty ! \n");
            mi_heap_destroy(thread_local_ex_mimalloc_heap);
            // _mi_heap_destroy_pages(thread_local_ex_mimalloc_heap);
        }
    }
}

// bool ex_mimalloc::heap_is_unused(void)
// {
//     if (!thread_local_ex_mimalloc_heap)
//     {
//         fmt::print("[warning] First time seeing this heap, cannot check it. \n");
//         return false;
//     }
//     return mi_heap_visit_blocks(thread_local_ex_mimalloc_heap, )
// }

// template<typename T>
// std::size_t ex_mimalloc::get_usable_size(T* ptr) {
//     if (!thread_local_ex_mimalloc_heap)
//     {
//         fmt::print("[warning] First time seeing this heap, cannot get its usable size. \n");
//         return 0;
//     } else
//     {
//          return mi_usable_size((void*) (ptr));
//     }
// }

// bool ex_mimalloc::block_is_unused(const mi_heap_t *heap, const mi_heap_area_t *area, void *block, size_t block_size, void *arg)
// {
//     if (!thread_local_ex_mimalloc_heap)
//     {
//         fmt::print("[warning] First time seeing this heap, cannot get its usable size. \n");
//         return false;
//     } else
//     {
//         if ( mi_usable_size(block) == block_size; ) { return true; }
//         else { return false; }
//     }
// }