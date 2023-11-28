#include <allocator.hpp>
#include <task_group.hpp>

#include <barrier>
#include <fmt/std.h>
#include <iostream>
#include <math.h>

/* TODO:
    - how to extend the arena size ?
    - numa node stuff, steal it from Fabian and get how to use it
    - for now, MI_OVERRIDE has to be set to OFF otherwise we can't use
      std::malloc ----> solved by using mmap
    - device stuff
      - get the device id in the device_memory constructor
      - check if ptr is actually on device for user_device_memory
    - RMA keys functions, the ones for individual objects (with offset)
    - UCX
    - MPI
    - concepts for Key, Register and Malloc
    - choose appropriate flags in the cuda pinning
    - choose appropriate flags in the libfabric backend methods
    - add the choice of which numa node one wants to allocate using std::malloc
    - set is_large as an option in the ex_mimalloc constructor (and hence in
   resource and resource_builder)
    - (the rest of the allocator class)
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
    std::size_t mem = 1ull << 29;
    const int nb_threads = 1;
    const int nb_allocs = 100000;
    /* Build resource and allocator via resource_builder */
//#define USE_ALLOC
#ifdef USE_ALLOC
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().register_memory().on_host(mem);
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<uint32_t, resource_t>;
    alloc_t a(rb);
    fmt::print("\n\n");

    {
        /* Fill an array through several threads and deallocate all on thread 0*/
        fill_array_multithread(nb_threads, nb_allocs, a);
    }
    // usual_alloc<uint32_t>(a);
#else
    {
        heap_per_thread(nb_threads, nb_allocs, mem);
    }
#endif

    fmt::print("\n\n");
    mi_collect(false);
    mi_stats_print(NULL);
}

/* Fill an array through several threads and deallocate all on thread 0*/
template <typename Alloc>
void fill_array_multithread(const int nb_threads, const int nb_allocs, Alloc a)
{
    std::vector<uint32_t*> ptrs(nb_threads * nb_allocs);
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
        fmt::print("{} : [error] ex_mimalloc failed to create the arena. \n", ptr);
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

    //  fmt::print("\n\n");
    //    mi_collect(true);
    //    mi_stats_print(NULL);
}
