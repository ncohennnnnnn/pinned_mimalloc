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

ext_mimalloc::ext_mimalloc(void* ptr, const std::size_t size, const int numa_node)
{
    if (size != 0)
    {
        /** @brief Create the ext_mimalloc arena
     *   @param exclusive allows allocations if specifically for this arena
     *   @param is_committed set to true
     *   @param is_large could be an option
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
            fmt::print("{} : Mimalloc arena created with id {} \n", ptr, m_arena_id);
        }
    }

#if USE_UNORDERED_MAP
    fmt::print("Hello from USE_UNORDERED_MAP \n");
#endif
#if USE_TL_VECTOR
    fmt::print("Hello from USE_TL_VECTOR \n");
#endif
}

template <typename Context>
ext_mimalloc::ext_mimalloc(const Context& C)
{
    ext_mimalloc{C.get_address(), C.get_size(), C.get_numa_node()};
}

void* ext_mimalloc::allocate(const std::size_t size, const std::size_t alignment)
{
    if (!heap_exists())
    {
        set_heap();
        fmt::print("ext_mimalloc:: New thread local heap {} with arena id {} \n",
            (void*) (get_heap()), m_arena_id);
    }

    void* rtn = nullptr;
    if (unlikely(alignment))
    {
        rtn = mi_heap_malloc_aligned(get_heap(), size, alignment);
    }
    else
    {
        rtn = mi_heap_malloc(get_heap(), size);
    }
    //    fmt::print("{} : Memory allocated with size {} from heap {} \n", rtn, size,
    //        (void*) (tl_ext_mimalloc_heaps));
    return rtn;
}

void* ext_mimalloc::reallocate(void* ptr, std::size_t size)
{
    if (!heap_exists())
    {
        fmt::print("ERROR!!! how can this happpen (in reallocate) \n");
        return nullptr;
    }
    return mi_heap_realloc(get_heap(), ptr, size);
}

void ext_mimalloc::deallocate(void* ptr, std::size_t size)
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

mi_arena_id_t ext_mimalloc::get_arena()
{
    return m_arena_id;
}

mi_heap_t* ext_mimalloc::get_heap()
{
#if USE_UNORDERED_MAP
    if (!heap_exists())
    {
        fmt::print("[error] thread with no heap. \n");
        return nullptr;
    }
    return tl_ext_mimalloc_heaps[m_arena_id];
#endif
#if USE_TL_VECTOR
    if (!heap_exists())
    {
        fmt::print("[error] thread with no heap. \n");
        return nullptr;
    }
    return m_heaps.get();
#endif
#if (!USE_UNORDERED_MAP && !USE_TL_VECTOR)
    fmt::print("[error] no heap threading option selected 1. \n");
    return nullptr;
#endif
}

void ext_mimalloc::set_heap()
{
#if USE_UNORDERED_MAP
    if (heap_exists())
    {
        fmt::print("[error] heap already esists. \n");
        return;
    }
    tl_ext_mimalloc_heaps[m_arena_id] = mi_heap_new_in_arena(m_arena_id);
#endif
#if USE_TL_VECTOR
    if (heap_exists())
    {
        fmt::print("[error] heap already esists. \n");
        return;
    }
    /* TODO: Maybe add a deleter function */
    m_heaps = indexed_tl_ptr<mi_heap_t>{[this]() { return mi_heap_new_in_arena(m_arena_id); },
        [](mi_heap_t* heap) {
            mi_heap_destroy(heap);
            mi_free(heap);
        }};
#endif
#if (!USE_UNORDERED_MAP && !USE_TL_VECTOR)
    fmt::print("[error] no heap threading option selected 2. \n");
#endif
}

bool ext_mimalloc::heap_exists()
{
#if USE_UNORDERED_MAP
    return tl_ext_mimalloc_heaps.contains(m_arena_id);
#endif
#if USE_TL_VECTOR
    return (m_heaps.get() != nullptr);
#endif
#if (!USE_UNORDERED_MAP && !USE_TL_VECTOR)
    fmt::print("[error] no heap threading option selected 3. \n");
    return false;
#endif
}

// bool ext_mimalloc::is_in_arena(void* ptr)
// {
//     ...
// }

ext_mimalloc::~ext_mimalloc()
{
    if (heap_exists())
    {
        if (get_heap()->page_count != 0)
        {
            fmt::print("Heap not empty : calling mi_heap_destroy \n");
            mi_heap_destroy(get_heap());
        }
        else
        {
            fmt::print("[error] Heap still alive at end of arena lifetime! \n");
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
