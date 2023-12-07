#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>

/* ---------------------------------------------------------------------------- */
template <typename T, typename AllocFunction, typename SetFunction>
bool fill_array_values(const int nb_arenas, const int nb_threads, const int nb_allocs,
    std::vector<T*>& ptrs, AllocFunction&& alloc_fn, SetFunction&& set_fn)
{
    try
    {
        ptrs.resize(nb_arenas * nb_threads * nb_allocs, nullptr);
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
        threads.clear(); /* jthreads join automatically */
    }
    catch (...)
    {
        return false;
    }

    fmt::print("Allocation finished\n");
    return true;
}

/* ---------------------------------------------------------------------------- */
template <typename T, typename GetFunction, typename FreeFunction>
bool check_array_values(const int nb_arenas, const int nb_threads, const int nb_allocs,
    const std::vector<T*>& ptrs, GetFunction&& get_fn, FreeFunction&& free_fn)
{
    bool ok = true;
    try
    {
        for (int i = 0; i < nb_arenas * nb_threads * nb_allocs; ++i)
        {
            int thread_id = i / (nb_arenas * nb_allocs);
            int arena_id = i / (nb_threads * nb_allocs);
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

/* ---------------------------------------------------------------------------- */
/* create an allocator using a custom arena then fill an array through several 
   threads and deallocate all on thread */
template <typename allocation_type>
bool test_host_allocator(
    const int nb_arenas, const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().on_host();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<allocation_type, resource_t>;
    std::vector<alloc_t> allocators;
    for (int i = 0; i < nb_arenas; ++i)
    {
        allocators.push_back(alloc_t{rb, mem});
    }
    //
    std::vector<allocation_type*> ptrs(nb_arenas * nb_threads * nb_allocs, nullptr);
    //
    auto alloc_fn = [&allocators](int arena_index) {
        return allocators[arena_index].allocate(sizeof(allocation_type));
    };
    auto set_fn = [](allocation_type* ptr, allocation_type temp) { *ptr = temp; };
    ok &= fill_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) { return *ptr; };
    auto free_fn = [&allocators](int arena_index, allocation_type* ptr) {
        allocators[arena_index].deallocate(ptr);
    };
    ok &= check_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, get_fn, free_fn);
    ptrs.clear();
    allocators.clear();
    return ok;
}

/* ---------------------------------------------------------------------------- */
/* Test mirror allocator using cudamalloc/cudafree */
template <typename allocation_type>
bool test_mirror_allocator(
    const int nb_arenas, const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().on_host_and_device();
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
    ok &= fill_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, alloc_fn, set_fn);
    //
    auto get_fn = [](allocation_type* ptr) {
        allocation_type temp{0};
        cudaMemcpy(&temp, ptr, sizeof(allocation_type), cudaMemcpyDeviceToHost);
        return temp;
    };
    auto free_fn = [&a](int /*arena_index*/, allocation_type* ptr) { a.deallocate(ptr); };
    ok &= check_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, get_fn, free_fn);
    ptrs.clear();
    return ok;
}

// /* ---------------------------------------------------------------------------- */
// /* Test mirror allocator using cudamalloc/cudafree for a single kind of mirror resource */
// template <typename allocation_type, typename Resource>
// bool test_mirror_allocator_rb(const resource_builder<Resource> rb, const int nb_arenas,
//     const int nb_threads, const int nb_allocs, std::size_t mem, void* ptr_a = nullptr,
//     void* ptr_b = nullptr)
// {
//     bool ok = true;
//     if (ptr_a){
//         if (ptr_b){
//             pmimallocator<allocation_type, Resource> a(rb, ptr_a, ptr_b, mem);
//         }
//     }
//     pmimallocator<allocation_type, Resource> a(rb, ptr_a, ptr_b, mem);
//     //
//     std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
//     //
//     auto alloc_fn = [&a](int /*arena_index*/) { return a.allocate(sizeof(allocation_type)); };
//     auto set_fn = [](allocation_type* ptr, allocation_type temp) {
//         cudaMemcpy(ptr, &temp, sizeof(allocation_type), cudaMemcpyHostToDevice);
//     };
//     ok &= fill_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, alloc_fn, set_fn);
//     //
//     auto get_fn = [](allocation_type* ptr) {
//         allocation_type temp{0};
//         cudaMemcpy(&temp, ptr, sizeof(allocation_type), cudaMemcpyDeviceToHost);
//         return temp;
//     };
//     auto free_fn = [&a](int /*arena_index*/, allocation_type* ptr) { a.deallocate(ptr); };
//     ok &= check_array_values(nb_arenas, nb_threads, nb_allocs, ptrs, get_fn, free_fn);
//     ptrs.clear();
//     return ok;
// }

// /* ---------------------------------------------------------------------------- */
// /* Test mirror allocator using cudamalloc/cudafree for all mirror resources */
// template <typename allocation_type>
// bool test_mirror_allocator(
//     const int nb_arenas, const int nb_threads, const int nb_allocs, std::size_t mem)
// {
//     bool ok[18];
//     resource_builder RB;
//     std::size_t i = 0;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().register().pin().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().register().cuda_pin().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().register().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().pin().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().cuda_pin().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     ok[i] = test_mirror_allocator_rb<allocation_type>(
//         RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//     ++i;
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;
//     }
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;
//     }
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;
//     }
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;
//     }
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;sbatch_file_alloc-test-new_jemalloc
//     }
//     {
//         host_memory<base> hm(mem);
//         ok[i] = test_mirror_allocator_rb<allocation_type>(
//             RB.clear().use_mimalloc().host_device(), nb_arenas, nb_allocs, mem);
//         ++i;
//     }
// }
