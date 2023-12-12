#include <cstdint>
#include <memory_resource>
#include <stdexcept>
#include <unistd.h>

class ext_stdmalloc : public std::pmr::monotonic_buffer_resource
{
public:
    ext_stdmalloc(void* ptr, const std::size_t size, const int numa_node = -1)
      : std::pmr::monotonic_buffer_resource{ptr, size}
    {
    }

    ~ext_stdmalloc() {}
};
