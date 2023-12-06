#include <barrier>
#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>
//
#include <cuda_runtime.h>

/* TODO:
    - Fix segfault
    - how to extend the arena size ? Currently we cannot grow a custom arena, mimallloc handles this
      internally, why can't we hook into it for our arenas 

    - make the mirroring more flexible (host/host, higher number of memory chunks 
      entangled) and make it modular

    - numa node stuff, steal it from Fabian and get how to use it

    - UCX
    - MPI

    - in ext_stdmalloc change std::malloc to a pmr::malloc on the context 
      (+ numa stuff ?)

    - Max arena sizes (s<<32), currently mmap fails if we ask for too much, does it change on daint,
      does it change if we use module load craype-hugepages1G

    - Max memory pinning - same problem, we can only pin so many pages, can we use huge pages to 
      improve the problem
    
    - Run test on daint, more threads, more memory, more graphs, huge pages, what do we learn ?
            
    - LRU Cache, registration of user memory "on the fly" would benefit from an LRU memory page 
      cache. Steal code from MPI if time permits.
*/

// void mi_heap_destroy(mi_heap_t* heap);
using unique_tls_heap = std::unique_ptr<mi_heap_t, void (*)(mi_heap_t*)>;
thread_local unique_tls_heap thread_local_heap_{nullptr, mi_heap_destroy};

mi_heap_t* create_tls_heap(mi_arena_id_t m_arena_id)
{
    return mi_heap_new_in_arena(m_arena_id);
}

void heap_per_thread(const int nb_threads, const int nb_allocs, std::size_t mem);

template <typename Alloc>
void fill_array_multithread(const int nb_threads, const int nb_allocs, Alloc a);

template <typename Alloc>
void std_vector(Alloc a);

template <typename Alloc>
void fill_buffer(Alloc a);

template <typename T, typename Alloc>
void usual_alloc(Alloc a);

int main()
{
    // minimum arena 25, maximum arena when pinning 30, maximum mmap 35
    std::size_t mem = 1ull << 35;
    const int nb_threads = 4;
    const int nb_allocs = 100000;

#define USE_ALLOC
    // #define USE_MIRROR

#ifdef USE_ALLOC
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin() /*.register_memory()*/.on_host();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<uint32_t, resource_t>;
    alloc_t a(rb, mem);
    alloc_t b(rb, mem);

    fmt::print("\n\n");

    {
        /* Fill an array through several threads and deallocate all on thread 0*/
        // fill_array_multithread(nb_threads, nb_allocs, a);
        // fill_array_multithread(nb_threads, nb_allocs, b);
    }
    // usual_alloc<uint32_t>(a);

#elif defined(USE_MIRROR)
    {
        mirror_memory<base> dm(mem);
        void* ptr = dm.get_address();
        mi_arena_id_t arena_id;
        int numa_node = -1;

        bool success =
            mi_manage_os_memory_ex(ptr, mem, true, false, true, numa_node, true, &arena_id);
        if (!success)
        {
            fmt::print("{} : [error] ext_mimalloc failed to create the arena. \n", ptr);
        }
        else
        {
            fmt::print("{} : Mimalloc arena created \n", ptr);
        }
        __device__ mi_heap_t* heap_ = mi_heap_new_in_arena(arena_id);
        // void* p1 = mi_heap_malloc(heap_, 32);
        // fmt::print("{} : device ptr allocated \n", p1);
        // void* p2 = mi_heap_malloc(heap_, 48);
        // fmt::print("{} : device ptr allocated \n", p2);
        // mi_free(p1);
        // mi_free(p2);
    }

#else
    {
        // heap_per_thread(nb_threads, nb_allocs, mem);
    }
#endif
    fmt::print("\n\n");
    // mi_collect(false);
    // mi_stats_print(NULL);
}

/* Fill an array through several threads and deallocate all on thread 0*/
template <typename Alloc>
void fill_array_multithread(const int nb_threads, const int nb_allocs, Alloc a)
{
    std::vector<uint32_t*> ptrs(nb_threads * nb_allocs);
    // uint32_t* ptrs[nb_threads * nb_allocs];
    std::vector<std::thread> threads;

    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(
            std::thread{[&a, &nb_allocs, &ptrs](int thread_id) mutable {
                            fmt::print("{}: {}, \n", thread_id, std::this_thread::get_id());

                            fmt::print("Thread {} \n", thread_id);
                            for (int i = 0; i < nb_allocs; ++i)
                            {
                                ptrs[thread_id * nb_allocs + i] = a.allocate(32);
                                *ptrs[thread_id * nb_allocs + i] = thread_id * nb_allocs + i;
                            }
                        },
                thread_id});
    }

    for (auto& t : threads)
        t.join();
    fmt::print("finished\n");

    fmt::print("Clearing memory \n");

    for (int i = 0; i < nb_allocs * nb_threads; ++i)
    {
        int thread_id = i / nb_allocs;
        if (*ptrs[i] == i)
        {
            a.deallocate(ptrs[i]);
        }
        else
        {
            fmt::print("[ERROR] from thread {}, expected {}, got {} \n", thread_id, i, *ptrs[i]);
        }
    }

    fmt::print("Checked ok\n");
    threads.clear();
    ptrs.clear();

    //  fmt::print("\n\n");
    //  mi_collect(false);
    //  mi_stats_print(NULL);
}

/* Standard vector */
template <typename Alloc>
void std_vector(Alloc a)
{
    fmt::print("Standard vector \n");
    // fmt::print("Resource use count : {} \n", res.use_count());
    std::vector<int, Alloc> v(100, a);
    // fmt::print("Resource use count : {} \n", res.use_count());
    fmt::print("{} : Vector data \n", (void*) v.data());
    for (std::size_t i; i < 100; ++i)
    {
        v[i] = 1;
    }
    for (std::size_t i; i < 100; ++i)
    {
        fmt::print("{}, ", v[i]);
    }
    fmt::print("\n\n");
}

/* Buffer filling */
template <typename Alloc>
void fill_buffer(Alloc a)
{
    fmt::print("Buffer filling\n");
    int* buffer[1000];
    for (std::size_t i; i < 100; ++i)
    {
        buffer[i] = a.allocate(8);
    }
    for (std::size_t i; i < 100; ++i)
    {
        a.deallocate(buffer[i]);
    }
    fmt::print("\n\n");
}

/* Usual allocation */
template <typename T, typename Alloc>
void usual_alloc(Alloc a)
{
    fmt::print("Usual allocation\n");
    T* p1 = a.allocate(32);
    T* p2 = a.allocate(48);
    a.deallocate(p1);
    a.deallocate(p2);
    fmt::print("\n\n");
}

/* Build an arena, then 1 heap per thread, allocate with them */
void heap_per_thread(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    // mi_option_set(mi_option_limit_os_alloc, 1);
    host_memory<base> hm(mem);
    void* ptr = hm.get_address();
    mi_arena_id_t m_arena_id{};

    mi_heap_t* heaps[nb_threads];

    std::vector<std::uint32_t*> ptrs(nb_threads * nb_allocs);
    bool success = mi_manage_os_memory_ex(ptr, mem, true, false, true, -1, true, &m_arena_id);
    if (!success)
    {
        fmt::print("{} : [error] ext_mimalloc failed to create the arena. \n", ptr);
    }
    else
    {
        fmt::print("{} : Mimalloc arena created \n", ptr);
    }
    fmt::print("\n");
    std::vector<std::thread> threads;

// #define USE_DODGY_TASK_LIB
#ifdef USE_DODGY_TASK_LIB
    {
        threading::task_system ts(nb_threads, true);
        threading::parallel_for::apply(
            nb_threads, &ts, [&heaps, m_arena_id, &nb_allocs, &ptrs](int thread_id) mutable {
                fmt::print("Thread Id " << std::this_thread::get_id() << std::endl;

                if (!thread_local_heap_)
                {
                    fmt::print("New heap on thread " << thread_id << std::endl;
                    auto my_delete = [](mi_heap_t* heap) {
                        fmt::print("NOT Deleting heap (it's safe) " << heap << std::endl;
                        // mi_heap_destroy(heap);
                    };
                    thread_local_heap_ =
                        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
                }
                for (int i = 0; i < nb_allocs; ++i)
                {
                ptrs[thread_id * nb_allocs + i] =
                    static_cast<uint32_t*>(mi_heap_malloc(thread_local_heap_.get(), 32));
                *ptrs[thread_id * nb_allocs + i] = thread_id * nb_allocs + i;
}
            });
    }
#else
    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(std::thread{
            [&heaps, m_arena_id, &nb_allocs, &ptrs](int thread_id) mutable {
                std::cout << thread_id << ": " << std::this_thread::get_id() << std::endl;
                // std::cout << _mi_thread_id() << std::endl;
                // heaps[thread_id] = mi_heap_new_in_arena(m_arena_id);
                if (!thread_local_heap_)
                {
                    fmt::print("New heap on thread {}\n", thread_id);
                    auto my_delete = [](mi_heap_t* heap) {
                        fmt::print("NOT Deleting heap (it's safe) {}\n", (void*) (heap));
                        // mi_heap_destroy(heap);
                    };
                    thread_local_heap_ =
                        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
                }

                for (int i = 0; i < nb_allocs; ++i)
                {
                    ptrs[thread_id * nb_allocs + i] =
                        static_cast<uint32_t*>(mi_heap_malloc(thread_local_heap_.get(), 32));
                    *ptrs[thread_id * nb_allocs + i] = thread_id * nb_allocs + i;
                }
            },
            thread_id});
    }
    for (auto& t : threads)
    {
        t.join();
    }
    fmt::print("finished\n");

    fmt::print("Clearing memory \n");

    for (int i = 0; i < nb_allocs * nb_threads; ++i)
    {
        int thread_id = i / nb_allocs;
        // fmt::print("{} \n", thread_id);
        if (*ptrs[i] == i)
        {
            mi_free(ptrs[i]);
        }
        else
        {
            fmt::print("[ERROR] from thread {}, expected {}, got {} \n", thread_id, i, *ptrs[i]);
        }
    }

    threads.clear();
#endif

    //  fmt::print("\n\n");
    //    mi_collect(true);
    //    mi_stats_print(NULL);
}
