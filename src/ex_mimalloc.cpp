#include <ex_mimalloc.hpp>
#include <task_group.hpp>

#include <fmt/core.h>


ex_mimalloc::ex_mimalloc(void* ptr, const std::size_t size, const int numa_node){
    if (size != 0) {
        /** @brief Create the ex_mimalloc arena
        * @param exclusive allows allocations if specifically for this arena
        * @param is_committed set to false
        * 
        * TODO: @param is_large could be an option
        */
        bool success = mi_manage_os_memory_ex(ptr, size, true, false, true, numa_node, true, &m_arena_id);
        if (!success) {
            fmt::print("{} : [error] ex_mimalloc failed to create the arena. \n", ptr);
        } else { fmt::print("{} : Mimalloc arena created \n", ptr); }

        /* Associate a heap per head to the arena */
        threading::task_system ts(nb_threads, true);
        threading::parallel_for::apply(nb_threads, &ts,
            [this, ptr](int thread_id) mutable 
            {
                // m_threads[thread_id] = std::make_pair(thread_id,get_thread_id());
                m_threads[thread_id] = std::make_pair(thread_id,std::this_thread::get_id());
                m_heap[thread_id] = mi_heap_new_in_arena(m_arena_id);
                if (m_heap[thread_id] == nullptr) {
                    fmt::print("{} : [error] ex_mimalloc failed to create the heap on thread {}. \n"
                                , ptr, thread_id);
                } else { std::cout << ptr << " : Mimalloc heap created on thread " << thread_id << ", " << std::this_thread::get_id() << "\n"; }
            }
        );

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
    // int thread = get_thread_id();
    std::thread::id thread_id = std::this_thread::get_id();
    int id = nb_threads;
    for(int i = 0; i<nb_threads; ++i){ 
        if (thread_id == std::get<1>(m_threads[i])) { id = i; }
    }
    if ( id == nb_threads) { std::cout << "[error] Wrong thread id : " << thread_id << "\n"; return nullptr; }
    if (unlikely(alignment)) {
        rtn = mi_heap_malloc_aligned(m_heap[id], size, alignment);
    } else {
        rtn = mi_heap_malloc(m_heap[id], size);
    }
    std::cout << rtn << " : Memory allocated with size " << size << " on thread " << thread_id << "\n";
    return rtn;
}

void* ex_mimalloc::reallocate(void* ptr, std::size_t size ) {
    // mi_threadid_t thread = get_thread_id();
    std::thread::id thread_id = std::this_thread::get_id();
    int id = nb_threads;
    for(int i = 0; i<nb_threads; ++i){ 
        if (thread_id == std::get<1>(m_threads[i])) { id = i; }
    }
    if ( id == nb_threads) { std::cout << "[error] Wrong thread id : "  << thread_id << "\n"; return nullptr; }
    return mi_heap_realloc(m_heap[id], ptr, size );
}

void ex_mimalloc::deallocate(void* ptr, std::size_t size ) {
    if (likely(ptr)) {
        if (unlikely(size)) { mi_free_size(ptr, size ); }
        else { mi_free(ptr); }
    }
    fmt::print("{} : Memory deallocated. \n", ptr);
}


// static inline void* get_prim_tls_slot(size_t slot) noexcept {
//   void* res;
//   const size_t ofs = (slot*sizeof(void*));
//   #if defined(__i386__)
//     __asm__("movl %%gs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  // x86 32-bit always uses GS
//   #elif defined(__APPLE__) && defined(__x86_64__)
//     __asm__("movq %%gs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  // x86_64 macOSX uses GS
//   #elif defined(__x86_64__) && (MI_INTPTR_SIZE==4)
//     __asm__("movl %%fs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  // x32 ABI
//   #elif defined(__x86_64__)
//     __asm__("movq %%fs:%1, %0" : "=r" (res) : "m" (*((void**)ofs)) : );  // x86_64 Linux, BSD uses FS
//   #elif defined(__arm__)
//     void** tcb; MI_UNUSED(ofs);
//     __asm__ volatile ("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r" (tcb));
//     res = tcb[slot];
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
//     // issue #384, #495: on the Bionic libc (Android), slot 1 is the thread id
//     // see: https://github.com/aosp-mirror/platform_bionic/blob/c44b1d0676ded732df4b3b21c5f798eacae93228/libc/platform/bionic/tls_defines.h#L86
//     return (uintptr_t)get_prim_tls_slot(1);
//   #else
//     // in all our other targets, slot 0 is the thread id
//     // glibc: https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/x86_64/nptl/tls.h
//     // apple: https://github.com/apple/darwin-xnu/blob/main/libsyscall/os/tsd.h#L36
//     return (uintptr_t)get_prim_tls_slot(0);
//   #endif
// }