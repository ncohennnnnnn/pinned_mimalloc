#include <barrier>
#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>
#include "test_functions.hpp"

int main()
{
    // minimum arena 25, maximum arena when pinning 30, maximum mmap 35
    std::size_t mem = 1ull << 29;
    const int nb_threads = 4;
    const int nb_allocs = 50000;

    // Fill an array through several threads and deallocate all on thread 0
    bool ok = true;
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "Testing heap per thread " << i << std::endl;
        ok &= heap_per_thread<int>(nb_threads, nb_allocs, mem);
    }
    ok &= heap_per_thread<std::uint64_t>(nb_threads, nb_allocs, mem);
    ok &= heap_per_thread<double>(nb_threads, nb_allocs, mem);

    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
