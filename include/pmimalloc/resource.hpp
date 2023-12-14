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
    using context_t = Context;
    using this_type = resource<context_t, Malloc>;
    using shared_this = std::shared_ptr<resource<context_t, Malloc>>;
    using malloc_t = Malloc;
    using key_t = context_t::key_t;

    resource() = default;

    resource(const this_type& r) noexcept = delete;

    template <typename... Args>
    resource(Args&&... args)
      : context_t{std::forward<Args>(args)...}
      , m_malloc{context_t::get_address(), context_t::get_size(), context_t::get_numa_node()}
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
