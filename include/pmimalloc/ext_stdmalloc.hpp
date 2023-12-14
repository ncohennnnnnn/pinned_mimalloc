#include <cstdint>
#include <memory_resource>
#include <stdexcept>
#include <unistd.h>

class ext_stdmalloc
{
public:
    ext_stdmalloc(void* ptr, const std::size_t size, int numa_node)
      : m_mbuffer{ptr, size}
      , m_spool{&m_mbuffer}
    {
    }

    ~ext_stdmalloc() {}

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        void* rtn = m_spool.allocate(size, alignment);
        return rtn;
    }

    void* reallocate(void* ptr, std::size_t size)
    {
        fmt::print("Sorry, no reallocate function \n");
        return nullptr;
    }

    void deallocate(void* ptr, std::size_t size = 0)
    {
        std::size_t s;
        if (size == 0)
            s = sizeof(ptr);
        else
            s = size;
        m_spool.deallocate(ptr, s);
    }
    // fmt::print("{} : Memory deallocated. \n", ptr);

private:
    std::pmr::monotonic_buffer_resource m_mbuffer;
    std::pmr::synchronized_pool_resource m_spool;
};
