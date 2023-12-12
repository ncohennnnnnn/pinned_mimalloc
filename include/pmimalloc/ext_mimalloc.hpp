#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include <fmt/core.h>

#include <mimalloc.h>
/* TODO: Put this under a debug option */
#include <mimalloc/internal.h>

#include <pmimalloc/indexed_tl_ptr.hpp>

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

class ext_mimalloc
{
public:
    ext_mimalloc() = default;

    ext_mimalloc(void* ptr, const std::size_t size, const int numa_node)
    {
        if (size != 0)
        {
            /** @brief Create the ext_mimalloc arena
     *   @param exclusive allows allocations if specifically for this arena
     *   @param is_committed set to true
     *   @param is_large could be an option
     */

            /* Do not use OS memory for allocation (but only pre-allocated arena). */
            // mi_option_set(mi_option_limit_os_alloc, 1);

            /* OS tag to assign to mimalloc'd memory. */
            // mi_option_enable(mi_option_os_tag);
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
            m_heaps =
                indexed_tl_ptr<mi_heap_t>{[this]() { return mi_heap_new_in_arena(m_arena_id); },
                    [](mi_heap_t* heap) {
                        // if (heap->page_count != 0)
                        //     fmt::print("Heap not empty \n");
                        fmt::print("Deleting heap \n");
                    }};
        }
    }

    template <typename Context>
    ext_mimalloc(const Context& C);

    ext_mimalloc(const ext_mimalloc& m) = delete;

    ~ext_mimalloc() = default;

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
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

    void* reallocate(void* ptr, std::size_t size)
    {
        return mi_heap_realloc(get_heap(), ptr, size);
    }

    void deallocate(void* ptr, std::size_t size)
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

    mi_arena_id_t get_arena()
    {
        return m_arena_id;
    }

    mi_heap_t* get_heap()
    {
        return m_heaps.get();
    }

    std::size_t required_alignment()
    {
        return MIMALLOC_SEGMENT_ALIGNED_SIZE;
    }

private:
    mi_arena_id_t m_arena_id{};
    mi_stats_t m_stats;
    indexed_tl_ptr<mi_heap_t> m_heaps;
};
