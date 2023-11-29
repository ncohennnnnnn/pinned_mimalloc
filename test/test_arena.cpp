#include <barrier>
#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>

using unique_tls_heap = std::unique_ptr<mi_heap_t, void (*)(mi_heap_t*)>;
thread_local unique_tls_heap thread_local_heap_{nullptr, mi_heap_destroy};

mi_heap_t* create_tls_heap(mi_arena_id_t m_arena_id)
{
    return mi_heap_new_in_arena(m_arena_id);
}

//template <typename Alloc>
//void std_vector(Alloc a);

//template <typename Alloc>
//void fill_buffer(Alloc a);

//template <typename T, typename Alloc>
//void usual_alloc(Alloc a);

///* Standard vector */
//template <typename Alloc>
//void std_vector(Alloc a)
//{
//    fmt::print("Standard vector \n");
//    // fmt::print("Resource use count : {} \n", res.use_count());
//    std::vector<int, Alloc> v(100, a);
//    // fmt::print("Resource use count : {} \n", res.use_count());
//    fmt::print("{} : Vector data \n", (void*) v.data());
//    for (std::size_t i; i < 100; ++i)
//    {
//        v[i] = 1;
//    }
//    for (std::size_t i; i < 100; ++i)
//    {
//        fmt::print("{}, ", v[i]);
//    }
//    fmt::print("\n\n");
//}

/* Buffer filling */
//template <typename Alloc>
//void fill_buffer(Alloc a)
//{
//    fmt::print("Buffer filling\n");
//    int* buffer[1000];
//    for (std::size_t i; i < 100; ++i)
//    {
//        buffer[i] = a.allocate(8);
//    }
//    for (std::size_t i; i < 100; ++i)
//    {
//        a.deallocate(buffer[i]);
//    }
//    fmt::print("\n\n");
//}

/* Usual allocation */
//template <typename T, typename Alloc>
//void usual_alloc(Alloc a)
//{
//    fmt::print("Usual allocation\n");
//    T* p1 = a.allocate(32);
//    T* p2 = a.allocate(48);
//    a.deallocate(p1);
//    a.deallocate(p2);
//    fmt::print("\n\n");
//}

// ----------------------------------------------------------------------------
// Fill an array through several threads and deallocate all on thread
bool test_allocator_threaded(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    using allocation_type = std::int64_t;
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().register_memory().on_host();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<allocation_type, resource_t>;
    alloc_t a(rb, mem);

    fmt::print("\n\n");
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    std::vector<std::thread> threads;

    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(
            std::thread{[&a, &nb_allocs, &ptrs](int thread_id) mutable {
                            fmt::print("{}: {}, \n", thread_id, std::this_thread::get_id());
                            for (int i = 0; i < nb_allocs; ++i)
                            {
                                allocation_type* ptr = a.allocate(sizeof(allocation_type));
                                ptrs[thread_id * nb_allocs + i] = ptr;
                                *ptr = thread_id * nb_allocs + i;
                            }
                        },
                thread_id});
    }

    for (auto& t : threads)
    {
        t.join();
    }
    fmt::print("finished\n");
    fmt::print("Checking memory \n");
    std::fflush(stdout);
    for (int i = 0; i < nb_allocs * nb_threads; ++i)
    {
        int thread_id = i / nb_allocs;
        if (*ptrs[i] == i)
        {
            a.deallocate(ptrs[i]);
        }
        else
        {
            ok = false;
            fmt::print("[ERROR] from thread {}, expected {}, got {} \n", thread_id, i, *ptrs[i]);
        }
    }

    fmt::print("Checked ok\n");
    threads.clear();
    ptrs.clear();
    return ok;
}

// ----------------------------------------------------------------------------
// create arenas, per thread, allocate with them, deallocate on thread 0
bool heap_per_thread(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    host_memory<base> hm(mem);
    void* ptr = hm.get_address();
    mi_arena_id_t m_arena_id{};

    std::vector<mi_heap_t*> heaps(nb_threads, nullptr);
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
            ok = false;
            fmt::print("[ERROR] from thread {}, expected {}, got {} \n", thread_id, i, *ptrs[i]);
        }
    }
    threads.clear();
    return ok;
}

int main()
{
    // minimum arena 25, maximum arena when pinning 30, maximum mmap 35
    std::size_t mem = 1ull << 29;
    const int nb_threads = 1;
    const int nb_allocs = 5;

    // Fill an array through several threads and deallocate all on thread 0
    bool ok = true;
    ok &= test_allocator_threaded(nb_threads, nb_allocs, mem);

    // ok &= heap_per_thread(nb_threads, nb_allocs, mem);

    // usual_alloc<uint32_t>(a);

    //  fmt::print("\n\n");
    //  mi_collect(false);
    //  mi_stats_print(NULL);

    // ctest 0 on exit indicates SUCCESS
    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
