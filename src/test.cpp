#include <allocator.hpp>
// #include <vector>

/* TODO:
    - single arena with size = size*nb_of_threads
    - numa node stuff, steal it from Fabian and get how to use it
    - for now, MI_OVERRIDE has to be set to OFF otherwise we can't use std::aligned_alloc,
      try to find a way to either write ur own aligned_alloc or keep it like this.
    - device stuff
        - get the device id in the device_memory constructor
        - check if ptr is actually on device for user_device_memory
    - RMA keys functions, the ones for individual objects (with offset)
    - UCX
    - MPI
    - concepts for Key, Register and Malloc
    - choose appropriate flags in the cuda pinning
    - choose appropriate flags in the libfabric backend methods
    - add the choice of which numa node one wants to allocate using std::malloc
    - set is_large as an option in the ex_mimalloc constructor (and hence in resource and resource_builder)
    - (the rest of the allocator class)
*/

int main() {



    std::size_t mem = 1 << 25;

    resource_builder rb;
    auto res = rb.use_mimalloc().pin().register_memory().on_host(mem);
    using resource_t = decltype(res.build());
    using alloc_t    = pmimallocator<int, resource_t>;
    alloc_t a(res.sbuild());
    fmt::print("\n\n");

/* Standard vector */
    fmt::print("Standard vector\n");
    std::vector<int, alloc_t> v(100,a);
    for(std::size_t i; i<100; ++i){
        v[i] = 1.0;
    }
    for(std::size_t i; i<100; ++i){
        fmt::print("{}, ", v[i]);
    }
    fmt::print("\n\n");

/* Usual allocation */
    fmt::print("Usual allocation\n");
    int* p1  = a.allocate(32);
    int* p2  = a.allocate(48);
    a.deallocate(p1);
    a.deallocate(p2);
    fmt::print("\n\n");

/* Buffer filling */
    // fmt::print("Buffer filling\n");
    // int* buffer[1000];
    // for(std::size_t i; i < 100; ++i){
    //     buffer[i] = a.allocate(8);
    // }
    // for(std::size_t i; i < 100; ++i){
    //     a.deallocate(buffer[i]);
    // }
    // fmt::print("\n\n");

    mi_collect(true);
    mi_stats_print(NULL);

    return 0;
}
