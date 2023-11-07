#include <allocator.hpp>

int main() {

    std::size_t mem = 1 << 25;
    resource_builder rb;
    rb.use_mimalloc().pin().register_memory().on_device(mem);
    pmimallocator<double, decltype(rb.build())> a(rb);

    double* p1  = a.allocate(32);
    double* p2  = a.allocate(48);

    a.deallocate(p1);
    a.deallocate(p2);

    mi_collect(true);
    mi_stats_print(NULL);

    return 0;
}
