#pragma once

#include <memory>
#include <tuple>

/*------------------------------------------------------------------*/
/*                            Resource                              */
/*------------------------------------------------------------------*/

template <typename Context, typename Malloc>
class resource : public Context
{
public:
    using this_type = resource<Context, Malloc>;
    using shared_this = std::shared_ptr<resource<Context, Malloc>>;
    using malloc_t = Malloc;
    using context_t = Context;

    resource() noexcept
      : m_malloc{}
    {
    }

    resource(context_t&& c) noexcept
      : context_t{std::move(c)}
      , m_malloc{&c}
    {
    }

    resource(const this_type& r) noexcept = delete;

    /* Should I do this instead of the next three ones ?*/
    // template<typename... Args>
    // resource(Args... args)
    //   : context_t{args...}
    //   , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    // {
    // }

    resource(const std::size_t size, const std::size_t alignment = 0)
      : context_t{size, alignment}
      , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    {
    }

    resource(void* ptr, const std::size_t size)
      : context_t{ptr, size}
      , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    {
    }

    resource(void* ptr_a, void* ptr_b, const std::size_t size)
      : context_t{ptr_a, ptr_b, size}
      , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    {
    }

#if (defined(WITH_MIMALLOC) && defined(USE_TL_VECTOR))
    void* allocate(const std::size_t size, const std::size_t idx, const std::size_t alignment = 0)
    {
        return m_malloc.allocate(size, idx, alignment);
    }
#else
    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return m_malloc.allocate(size, alignment);
    }
#endif

    void* reallocate(void* ptr, const std::size_t size)
    {
        return m_malloc.reallocate(ptr, size);
    }

    void deallocate(void* ptr, const std::size_t size = 0)
    {
        return m_malloc.deallocate(ptr, size);
    }

protected:
    malloc_t m_malloc;
};

/*------------------------------------------------------------------*/
/*                          No allocator                            */
/*------------------------------------------------------------------*/

class ext_none
{
};