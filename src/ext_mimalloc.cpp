#include <unordered_map>

#include <pmimalloc/ext_mimalloc.hpp>

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
// thread_local mi_heap_t* tl_ext_mimalloc_heap{nullptr};
thread_local std::unordered_map<mi_arena_id_t, mi_heap_t*> tl_ext_mimalloc_heaps;

ext_mimalloc::ext_mimalloc(void* ptr, const std::size_t size, const int numa_node)
{
    if (size != 0)
    {
        /** @brief Create the ext_mimalloc arena
     * @param exclusive allows allocations if specifically for this arena
     * @param is_committed set to true
     * @param is_large could be an option
     */

        /* Do not use OS memory for allocation (but only pre-allocated arena). */
        mi_option_set(mi_option_limit_os_alloc, 1);

        /* OS tag to assign to mimalloc'd memory. */
        mi_option_enable(mi_option_os_tag);
        bool success =
            mi_manage_os_memory_ex(ptr, size, true, false, true, numa_node, true, &m_arena_id);
        if (!success)
        {
            fmt::print("{} : [error] ext_mimalloc failed to create the arena. \n", ptr);
        }
        else
        {
            fmt::print("{} : Mimalloc arena created \n", ptr);
        }

        if (!tl_ext_mimalloc_heaps.contains(m_arena_id))
        {
            // auto my_delete = [](mi_heap_t* heap) {
            //     fmt::print("ext_mimalloc:: NOT Deleting heap (it's safe) {}\n", (void*) (heap));
            //     mi_heap_destroy(heap);
            // };
            tl_ext_mimalloc_heaps[m_arena_id] = mi_heap_new_in_arena(m_arena_id);
            //        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
            fmt::print("ext_mimalloc:: New thread local backing heap {} ",
                (void*) (tl_ext_mimalloc_heaps[m_arena_id]));
        }
        // mi_heap_set_default(tl_ext_mimalloc_heaps[m_arena_id]);
    }
}

template <typename Context>
ext_mimalloc::ext_mimalloc(const Context& C)
{
    ext_mimalloc{C.get_address(), C.get_size(), C.get_numa_node()};
}

void* ext_mimalloc::allocate(const std::size_t size, const std::size_t alignment)
{
    if (!tl_ext_mimalloc_heaps.contains(m_arena_id))
    {
        // auto my_delete = [](mi_heap_t* heap) {
        //     fmt::print("ext_mimalloc:: NOT Deleting heap (it's safe) {}\n", (void*) (heap));
        //     mi_heap_destroy(heap);
        // };
        tl_ext_mimalloc_heaps[m_arena_id] = mi_heap_new_in_arena(m_arena_id);
        //        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
        fmt::print("ext_mimalloc:: New thread local backing heap {} ",
            (void*) (tl_ext_mimalloc_heaps[m_arena_id]));
    }

    void* rtn = nullptr;
    if (unlikely(alignment))
    {
        rtn = mi_heap_malloc_aligned(tl_ext_mimalloc_heaps[m_arena_id], size, alignment);
    }
    else
    {
        rtn = mi_heap_malloc(tl_ext_mimalloc_heaps[m_arena_id], size);
    }
    //    fmt::print("{} : Memory allocated with size {} from heap {} \n", rtn, size,
    //        (void*) (tl_ext_mimalloc_heaps));
    return rtn;
}

void* ext_mimalloc::reallocate(void* ptr, std::size_t size)
{
    if (!tl_ext_mimalloc_heaps.contains(m_arena_id))
    {
        fmt::print("ERROR!!! how can this happpen \n");
    }
    return mi_heap_realloc(tl_ext_mimalloc_heaps[m_arena_id], ptr, size);
}

void ext_mimalloc::deallocate(void* ptr, std::size_t size)
{
    // uintptr_t ptr_ = reinterpret_cast<uintptr_t>(ptr);
    // uintptr_t heap_ = reinterpret_cast<uintptr_t>(mi_heap_get_backing());
    // ptrdiff_t diff = ptr_ - heap_;
    // if (diff > 0)
    //     fmt::print("BIG PROBLEM \n");
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

mi_heap_t* ext_mimalloc::get_heap()
{
    if (!tl_ext_mimalloc_heaps.contains(m_arena_id))
    {
        fmt::print("[error] thread with no heap. \n");
        return NULL;
    }
    return tl_ext_mimalloc_heaps[m_arena_id];
}

ext_mimalloc::~ext_mimalloc()
{
    if (!tl_ext_mimalloc_heaps.contains(m_arena_id))
    {
        fmt::print("ERROR!!! how can this happpen \n");
    }
    else
    {
        if (tl_ext_mimalloc_heaps[m_arena_id]->page_count != 0)
        {
            fmt::print("Heap not empty ! \n");
            mi_heap_destroy(tl_ext_mimalloc_heaps[m_arena_id]);
            // _mi_heap_destroy_pages(tl_ext_mimalloc_heaps[m_arena_id]);
        }
    }
}

// template<typename T>
// std::size_t ext_mimalloc::get_usable_size(T* ptr) {
//     if (!tl_ext_mimalloc_heaps)
//     {
//         fmt::print("[warning] First time seeing this heap, cannot get its usable size. \n");
//         return 0;
//     } else
//     {
//          return mi_usable_size((void*) (ptr));
//     }
// }
