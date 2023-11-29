#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>

template <typename T, typename GetFunction, typename FreeFunction>
bool check_array_values(int nb_arenas, int nb_allocs, int nb_threads, const std::vector<T*>& ptrs,
    GetFunction&& get_fn, FreeFunction&& free_fn)
{
    bool ok = true;
    try
    {
        for (int i = 0; i < nb_allocs * nb_arenas * nb_threads; ++i)
        {
            int thread_id = i / (nb_allocs * nb_arenas);
            int arena_id = i / (nb_allocs * nb_threads);
            T temp = get_fn(ptrs[i]);
            if (temp == i)
            {
                free_fn(arena_id, ptrs[i]);
            }
            else
            {
                ok = false;
                fmt::print("[ERROR] from thread {} and arena {}, expected {}, got {} \n", thread_id,
                    arena_id, i, temp);
            }
        }
    }
    catch (...)
    {
        ok = false;
    }
    fmt::print("Memcheck finished : {}\n", ok);
    return ok;
}

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

    std::vector<mi_heap_t*> heaps(nb_threads, nullptr);
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    std::vector<std::jthread> threads;

    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(std::jthread{
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
    threads.clear();    // jthreads join automatically
    fmt::print("Allocation finished\n");

    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [](int /*alloc_index*/, allocation_type* ptr) { mi_free(ptr); };
    ok = check_array_values(1, nb_allocs, nb_threads, ptrs, get_fn, free_fn);
    ptrs.clear();
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

    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    std::vector<std::jthread> threads;

    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(
            std::jthread{[&a, &nb_allocs, &ptrs](int thread_id) mutable {
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

    threads.clear();    // jthreads join automatically
    fmt::print("Allocation finished\n");

    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [&a](int /*alloc_index*/, allocation_type* ptr) { a.deallocate(ptr); };
    ok = check_array_values(1, nb_allocs, nb_threads, ptrs, get_fn, free_fn);
    ptrs.clear();
    return ok;
}

// ----------------------------------------------------------------------------
// create an allocator using a custom arena then
// fill an array through several threads and deallocate all on thread
template <typename allocation_type>
bool test_allocator_threaded_multiarena(
    const int nb_arenas, const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().register_memory().on_host();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<allocation_type, resource_t>;
    std::vector<alloc_t> allocators;
    for (int i = 0; i < nb_arenas; ++i)
    {
        allocators.push_back(alloc_t{rb, mem});
    }

    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs * nb_arenas, nullptr);
    std::vector<std::jthread> threads;
    /* for(threads){for(arenas){for(allocs){...}}} */
    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(std::jthread{
            [&allocators, &nb_arenas, &nb_allocs, &ptrs](int thread_id) mutable {
                // fmt::print("Thread ({}, {}) ", thread_id, std::this_thread::get_id());
                // fmt::print("arena {} \n", j);
                for (int i = 0; i < nb_allocs; ++i)
                {
                    for (int j = 0; j < nb_arenas; ++j)
                    {
                        allocation_type* ptr = allocators[j].allocate(sizeof(allocation_type));
                        ptrs[thread_id * nb_arenas * nb_allocs + j * nb_allocs + i] = ptr;
                        *ptr =
                            allocation_type(thread_id * nb_allocs * nb_arenas + j * nb_allocs + i);
                    }
                }
            },
            thread_id});
    }
    threads.clear();    // jthreads join automatically
    fmt::print("Allocation finished\n");

    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [&allocators](int alloc_index, allocation_type* ptr) {
        allocators[alloc_index].deallocate(ptr);
    };
    ok = check_array_values(1, nb_allocs, nb_threads, ptrs, get_fn, free_fn);
    ptrs.clear();
    allocators.clear();
    return ok;
}
