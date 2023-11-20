#include <allocator.hpp>
#include <task_group.hpp>

#include <math.h>
#include <barrier>

/* TODO:
    - single arena with big size or one arena per thread ?
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

    std::size_t mem = 1ull << 30;

/* Build resource and allocator via resource_builder */
    resource_builder RB;
    auto rb = RB.use_mimalloc().pin()/*.register_memory()*/.on_host(mem);
    using resource_t = decltype(rb.build());
    using alloc_t    = pmimallocator<int, resource_t>;
    alloc_t a(rb);
    fmt::print("\n\n");

/* Build resource and allocator by hand */
    // using resource_t = resource <context <pinned <host_memory <base>> , backend> , ex_mimalloc> ;
    // using alloc_t = pmimallocator<int, resource_t>;
    // auto res = std::make_shared<resource_t>(mem);
    // alloc_t a(res);
    // fmt::print("\n");

/* Fill an array through several threads and deallocate all on thread 0*/
    const int num_threads = 2;
    const int nb_rep      = 5;
    int* p[num_threads*nb_rep];
    std::barrier sync_point(num_threads);
    threading::task_system ts(num_threads, true);
    threading::parallel_for::apply(num_threads, &ts,
        [a,&p, &sync_point, &nb_rep](int thread_id) mutable 
        {
            fmt::print("Thread {} \n", thread_id);
            int idx;
            for(std::size_t i = 0; i<nb_rep; ++i){
                idx = nb_rep*thread_id + i;
                p[idx] = a.allocate(2);
                fmt::print("{} : ptr allocated \n", static_cast<void*>(p[idx]));
                *p[idx] = idx;
            }
            sync_point.arrive_and_wait();
        }
    );
    // sync_point.arrive_and_wait();
    for(int i = 0; i < nb_rep-1; ++i){
        if(p[i+1]-p[i]!=1){
            a.deallocate(p[i]);
        } else { fmt::print("[ERROR] from thread {} \n", floor(i/nb_rep)); }
    }
    a.deallocate(p[nb_rep]);
    fmt::print("\n\n");

    // mi_collect(true);
    // mi_stats_print(NULL);

    return 0;
}

/* Standard vector */
    // fmt::print("Standard vector \n");
    // // fmt::print("Resource use count : {} \n", res.use_count());
    // std::vector<int, alloc_t> v(100,a);
    // // fmt::print("Resource use count : {} \n", res.use_count());
    // fmt::print("{} : Vector data \n", (void*)v.data());
    // for(std::size_t i; i<100; ++i){
    //     v[i] = 1;
    // }
    // for(std::size_t i; i<100; ++i){
    //     fmt::print("{}, ", v[i]);
    // }
    // fmt::print("\n\n");

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

/* Usual allocation */
    // fmt::print("Usual allocation\n");
    // int* p1  = a.allocate(32);
    // int* p2  = a.allocate(48);
    // a.deallocate(p1);
    // a.deallocate(p2);
    // fmt::print("\n\n");