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
    std::size_t mem = 1ull << 27;
    const int nb_arenas = 4;
    const int nb_threads = 3;
    const int nb_allocs = 2;
    bool ok = true;

    ok &= test_allocator_threaded_multiarena<int>(nb_arenas, nb_threads, nb_allocs, mem);

    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
