#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>

// ----------------------------------------------------------------------------
template <typename T, typename AllocFunction, typename SetFunction>
bool fill_array_values(const int nb_threads, const int nb_arenas, const int nb_allocs,
    std::vector<T*>& ptrs, AllocFunction&& alloc_fn, SetFunction&& set_fn)
{
    try
    {
        ptrs.resize(nb_threads * nb_arenas * nb_allocs, nullptr);
        std::vector<std::jthread> threads;
        for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
        {
            threads.push_back(std::jthread{
                [nb_arenas, nb_allocs, alloc_fn, set_fn, &ptrs](int thread_id) mutable {
                    for (int i = 0; i < nb_allocs; ++i)
                    {
                        for (int j = 0; j < nb_arenas; ++j)
                        {
                            T* ptr = alloc_fn(j);
                            ptrs[thread_id * nb_arenas * nb_allocs + j * nb_allocs + i] = ptr;
                            set_fn(ptr, T{thread_id * nb_allocs * nb_arenas + j * nb_allocs + i});
                        }
                    }
                },
                thread_id});
        }
        threads.clear();    // jthreads join automatically
    }
    catch (...)
    {
        return false;
    }

    fmt::print("Allocation finished\n");
    return true;
}

// ----------------------------------------------------------------------------
template <typename T, typename GetFunction, typename FreeFunction>
bool check_array_values(const int nb_threads, const int nb_arenas, const int nb_allocs,
    const std::vector<T*>& ptrs, GetFunction&& get_fn, FreeFunction&& free_fn)
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
    //
    auto alloc_fn = [m_arena_id](int /*arena_index*/) {
        if (!thread_local_heap_)
        {
            fmt::print("New heap on thread {}\n", std::this_thread::get_id());
            auto my_delete = [](mi_heap_t* heap) {
                fmt::print("NOT Deleting heap (it's safe) {}\n", (void*) (heap));
                // mi_heap_collect(heap, 0);
                // mi_heap_destroy(heap);
            };
            thread_local_heap_ = unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
        }
        return static_cast<allocation_type*>(
            mi_heap_malloc(thread_local_heap_.get(), sizeof(allocation_type)));
    };
    auto set_fn = [](allocation_type* ptr, allocation_type temp) { *ptr = temp; };
    ok &= fill_array_values(nb_threads, 1, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [](int /*arena_index*/, allocation_type* ptr) { mi_free(ptr); };
    ok &= check_array_values(nb_threads, 1, nb_allocs, ptrs, get_fn, free_fn);
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
    //
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    //
    auto alloc_fn = [&a](int /*arena_index*/) { return a.allocate(sizeof(allocation_type)); };
    auto set_fn = [](allocation_type* ptr, allocation_type temp) { *ptr = temp; };
    ok &= fill_array_values(nb_threads, 1, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [&a](int /*arena_index*/, allocation_type* ptr) { a.deallocate(ptr); };
    ok &= check_array_values(nb_threads, 1, nb_allocs, ptrs, get_fn, free_fn);
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
    //
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs * nb_arenas, nullptr);
    //
    auto alloc_fn = [&allocators](int arena_index) {
        return allocators[arena_index].allocate(sizeof(allocation_type));
    };
    auto set_fn = [](allocation_type* ptr, allocation_type temp) { *ptr = temp; };
    ok &= fill_array_values(nb_threads, nb_arenas, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [&allocators](int arena_index, allocation_type* ptr) {
        allocators[arena_index].deallocate(ptr);
    };
    ok &= check_array_values(nb_threads, nb_arenas, nb_allocs, ptrs, get_fn, free_fn);
    ptrs.clear();
    allocators.clear();
    return ok;
}

// ----------------------------------------------------------------------------
// Test mirror allocator using cudamalloc/cudafree
template <typename allocation_type>
bool test_mirror_allocator_threaded(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    // Build resource and allocator via resource_builder
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().register_memory().on_mirror();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<allocation_type, resource_t>;
    alloc_t a(rb, mem);
    //
    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    //
    auto alloc_fn = [&a](int /*arena_index*/) { return a.allocate(sizeof(allocation_type)); };
    auto set_fn = [](allocation_type* ptr, allocation_type temp) {
        cudaMemcpy(ptr, &temp, sizeof(allocation_type), cudaMemcpyHostToDevice);
    };
    ok &= fill_array_values(nb_threads, 1, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) {
        allocation_type temp{0};
        cudaMemcpy(&temp, ptr, sizeof(allocation_type), cudaMemcpyDeviceToHost);
        return temp;
    };
    auto free_fn = [&a](int /*arena_index*/, allocation_type* ptr) { a.deallocate(ptr); };
    ok &= check_array_values(nb_threads, 1, nb_allocs, ptrs, get_fn, free_fn);
    ptrs.clear();
    return ok;
}
