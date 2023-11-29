#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>

// ----------------------------------------------------------------------------
// create arena, manually create 1 heap per thread, then
// fill an array through several threads and deallocate all on thread
using unique_tls_heap = std::unique_ptr<mi_heap_t, void (*)(mi_heap_t*)>;
thread_local unique_tls_heap thread_local_heap_{nullptr, mi_heap_destroy};

template <typename allocation_type = std::int64_t>
bool heap_per_thread(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    host_memory<base> hm(mem);
    void* base_ptr = hm.get_address();
    mi_arena_id_t m_arena_id{};
    bool success = mi_manage_os_memory_ex(base_ptr, mem, true, false, true, -1, true, &m_arena_id);
    if (!success)
    {
        fmt::print("{} : [error] ext_mimalloc failed to create the arena. \n", base_ptr);
        return false;
    }
    else
    {
        fmt::print("{} : Mimalloc arena created \n", base_ptr);
    }
    fmt::print("\n");

    std::vector<mi_heap_t*> heaps(nb_threads, nullptr);
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
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
                        mi_heap_collect(heap, 0);
                        // mi_heap_destroy(heap);
                    };
                    thread_local_heap_ =
                        unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
                }
                fmt::print("{}: {}, \n", thread_id, std::this_thread::get_id());
                for (int i = 0; i < nb_allocs; ++i)
                {
                    allocation_type* ptr = static_cast<allocation_type*>(
                        mi_heap_malloc(thread_local_heap_.get(), sizeof(allocation_type)));
                    ptrs[thread_id * nb_allocs + i] = ptr;
                    *ptr = allocation_type(thread_id * nb_allocs + i);
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

// ----------------------------------------------------------------------------
// create an allocator using a custom arena then
// fill an array through several threads and deallocate all on thread
template <typename allocation_type>
bool test_allocator_threaded(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
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
                                *ptr = allocation_type(thread_id * nb_allocs + i);
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
