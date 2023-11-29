#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>
#include "test_functions.hpp"

// ----------------------------------------------------------------------------
// create an allocator using a custom arena then
// fill an array through several threads and deallocate all on thread
template <typename allocation_type>
bool test_mirror_allocator_threaded(const int nb_threads, const int nb_allocs, std::size_t mem)
{
    bool ok = true;
    /* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin().register_memory().on_mirror();
    using resource_t = decltype(rb.build());
    using alloc_t = pmimallocator<allocation_type, resource_t>;
    alloc_t a(rb, mem);

    std::vector<allocation_type*> ptrs(nb_threads * nb_allocs, nullptr);
    std::vector<std::thread> threads;

    for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id)
    {
        threads.push_back(std::thread{
            [&a, &nb_allocs, &ptrs](int thread_id) mutable {
                fmt::print("{}: {}, \n", thread_id, std::this_thread::get_id());
                for (int i = 0; i < nb_allocs; ++i)
                {
                    allocation_type* ptr = a.allocate(sizeof(allocation_type));
                    ptrs[thread_id * nb_allocs + i] = ptr;
                    allocation_type temp{thread_id * nb_allocs + i};
                    cudaMemcpy(ptr, &temp, sizeof(allocation_type), cudaMemcpyHostToDevice);
                }
            },
            thread_id});
    }

    for (auto& t : threads)
        t.join();
    threads.clear();
    fmt::print("Allocation finished\n");

    auto get_fn = [](allocation_type* ptr) {
        allocation_type temp{0};
        cudaMemcpy(&temp, ptr, sizeof(allocation_type), cudaMemcpyDeviceToHost);
        return temp;
    };
    auto free_fn = [&a](int /*alloc_index*/, allocation_type* ptr) { a.deallocate(ptr); };

    ok = check_array_values(1, nb_allocs, nb_threads, ptrs, get_fn, free_fn);
    ptrs.clear();
    return ok;
}

int main()
{
    // minimum arena 25, maximum arena when pinning 30, maximum mmap 35
    std::size_t mem = 1ull << 29;
    const int nb_threads = 4;
    const int nb_allocs = 50000;

    // Fill an array through several threads and deallocate all on thread 0
    bool ok = true;
    std::cout << "Testing allocator threaded " << std::endl;
    ok &= test_mirror_allocator_threaded<int>(nb_threads, nb_allocs, mem);

    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
