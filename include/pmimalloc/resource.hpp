#pragma once

#include <memory>
#include <tuple>

/*------------------------------------------------------------------*/
/*                            resource                              */
/*------------------------------------------------------------------*/

template <typename Context, typename Malloc>
class resource : public Context
{
public:
    using this_type = resource<Context, Malloc>;
    using shared_this = std::shared_ptr<resource<Context, Malloc>>;
    using malloc_t = Malloc;
    using context_t = Context;
    using key_t = context_t::key_t;

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

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return m_malloc.allocate(size, alignment);
    }

    void* reallocate(void* ptr, const std::size_t size)
    {
        return m_malloc.reallocate(ptr, size);
    }

    void deallocate(void* ptr, const std::size_t size = 0)
    {
        return m_malloc.deallocate(ptr, size);
    }

    template <typename T>
    key_t get_key(T* ptr)
    {
        return this->template get_key(ptr);
    }

protected:
    malloc_t m_malloc;
};

/*------------------------------------------------------------------*/
/*               Non-mirrored version of resource                   */
/*------------------------------------------------------------------*/

template <typename Resource>
class simple : public Resource
{
public:
    using resource_t = Resource;
    using this_type = simple<resource_t>;

    simple()
      : resource_t{}
    {
    }

    simple(const std::size_t size, const std::size_t alignment = 0)
      : resource_t{size, alignment}
    {
    }

    simple(void* ptr, const std::size_t size)
      : resource_t{ptr, size}
    {
    }

    simple(void* ptr_a, void* ptr_b, const std::size_t size)
      : resource_t{ptr_a, ptr_b, size}
    {
    }
};
