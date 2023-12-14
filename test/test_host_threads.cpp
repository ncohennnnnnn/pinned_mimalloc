#include <iostream>
#include <math.h>
//
#include <fmt/std.h>
//
#include <pmimalloc/allocator.hpp>
#include "test_functions.hpp"

int main()
{
    /*
Maximum size for mem : 35 (max of mmap on my machine)
Maximum size when pinning : 30
    */
    std::size_t mem = 1ull << 27;
    const int nb_arenas = 1;
    const int nb_threads = 4;
    const int nb_allocs = 5;
    bool ok = true;

    ok &= test_host_allocator<int>(nb_arenas, nb_threads, nb_allocs, mem);

    return (ok == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
