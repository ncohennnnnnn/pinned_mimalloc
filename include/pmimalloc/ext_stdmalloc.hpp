#include <cstdint>
#include <stdexcept>
#include <unistd.h>

// TODO: add NUMA-aware allocation ?
class ext_stdmalloc
{
public:
    ext_stdmalloc() noexcept = default;

    ext_stdmalloc(void* /*ptr*/, const std::size_t /*size*/, const int /*numa_node*/) {}

    template <typename Context>
    ext_stdmalloc(const Context& C)
    {
    }

    ext_stdmalloc(const ext_stdmalloc& m) = delete;

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return std::malloc(size);
    }

    void* reallocate(void* ptr, std::size_t size)
    {
        return std::realloc(ptr, size);
    }

    void deallocate(void* ptr, std::size_t size = 0)
    {
        std::free(ptr);
    }
};
