#include <cstdint>
#include <memory_resource>
#include <stdexcept>
#include <unistd.h>

class ext_stdmalloc /*: public std::pmr::synchronized_pool_resource*/
{
    // public:
    //     ext_stdmalloc(void* ptr, const std::size_t size)
    //     {
    //         std::pmr::monotonic_buffer_resource res(ptr, size);
    //         this
    //         {
    //             res
    //         }
    //     }

    //     ~ext_stdmalloc() {}
};
