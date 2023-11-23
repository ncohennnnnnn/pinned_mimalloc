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
     * @param is_committed set to false
     *
     * TODO: @param is_large could be an option
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
        // mi_option_set(mi_option_limit_os_alloc, 1);
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
        auto my_delete = [](mi_heap_t* heap) {
            fmt::print("ex_mimalloc:: NOT Deleting heap (it's safe) {}\n", (void*) (heap));
            // mi_heap_destroy(heap);
        };
        thread_local_ex_mimalloc_heap = mi_heap_new_in_arena(m_arena_id);
        //        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
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
        std::cout << "ERROR!!! how can this happpen" << std::endl;
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

// static inline void* get_prim_tls_slot(size_t slot) noexcept {
//   void* res;
//   const size_t ofs = (slot*sizeof(void*));
//   #if defined(__i386__)
//     __asm__("movl %%gs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  //
//     x86 32-bit always uses GS
//   #elif defined(__APPLE__) && defined(__x86_64__)
//     __asm__("movq %%gs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  //
//     x86_64 macOSX uses GS
//   #elif defined(__x86_64__) && (MI_INTPTR_SIZE==4)
//     __asm__("movl %%fs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  //
//     x32 ABI
//   #elif defined(__x86_64__)
//     __asm__("movq %%fs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  //
//     x86_64 Linux, BSD uses FS
//   #elif defined(__arm__)
//     void** tcb; MI_UNUSED(ofs);
//     __asm__ volatile ("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r"
//     (tcb)); res = tcb[slot];
//   #elif defined(__aarch64__)
//     void** tcb; MI_UNUSED(ofs);
//     #if defined(__APPLE__) // M1, issue #343
//     __asm__ volatile ("mrs %0, tpidrro_el0\nbic %0, %0, #7" : "=r" (tcb));
//     #else
//     __asm__ volatile ("mrs %0, tpidr_el0" : "=r" (tcb));
//     #endif
//     res = tcb[slot];
//   #endif
//   return res;
// }

// /* Original name in mimalloc is mi_prim_thread_id*/
// mi_threadid_t _mi_thread_id(void) noexcept {
//   #if defined(__BIONIC__)
//     // issue #384, #495: on the Bionic libc (Android), slot 1 is the thread
//     id
//     // see:
//     https://github.com/aosp-mirror/platform_bionic/blob/c44b1d0676ded732df4b3b21c5f798eacae93228/libc/platform/bionic/tls_defines.h#L86
//     return (uintptr_t)get_prim_tls_slot(1);
//   #else
//     // in all our other targets, slot 0 is the thread id
//     // glibc:
//     https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/x86_64/nptl/tls.h
//     // apple:
//     https://github.com/apple/darwin-xnu/blob/main/libsyscall/os/tsd.h#L36
//     return (uintptr_t)get_prim_tls_slot(0);
//   #endif
// }
