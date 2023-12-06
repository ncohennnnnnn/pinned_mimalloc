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
    const int nb_arenas = 1;
    const int nb_threads = 8;
    const int nb_allocs = 1000;

    // Fill an array through several threads and deallocate all on thread 0
    bool ok = true;
    std::cout << "Testing allocator threaded " << std::endl;
    ok &= test_mirror_allocator<int>(nb_arenas, nb_threads, nb_allocs, mem);

    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
