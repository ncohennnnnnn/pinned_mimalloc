#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "indexed_tl_ptr.hpp"
#include "resource.hpp"

template <typename T, typename Resource>
class pmimallocator
{
public:
    /* Types */
    // using base = pmimallocator<T, Resource>;
    using this_type = pmimallocator<T, Resource>;
    using resource_type = Resource;
    using shared_resource = std::shared_ptr<Resource>;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template <typename U>
    struct rebind
    {
        using other = pmimallocator<U, Resource>;
    };

    /* Contructors */
    pmimallocator() noexcept = default;
    pmimallocator(const pmimallocator& other) noexcept = default;
    pmimallocator(pmimallocator&&) noexcept = default;
    /* Construct from a shared_ptr */
    pmimallocator(shared_resource r) noexcept
      : m_sptr_resource{r}
    {
    }
    /* Construct from a resource_builder */
    template <typename Args>
    pmimallocator(const resource_builder<resource_type, Args>& rb) noexcept
      : m_sptr_resource{rb.sbuild()}
    {
    }
    /* Construct from a resource */
    pmimallocator(const resource_type& r) noexcept
      : m_sptr_resource{std::make_shared<resource_type>(r)}
    {
    }

    pmimallocator& operator=(const pmimallocator&) noexcept = default;
    pmimallocator& operator=(pmimallocator&&) noexcept = default;

    pmimallocator select_on_container_copy_construction()
    {
        return *this;
    }

    template <class U>
    pmimallocator(const pmimallocator<U, Resource>& other)
      : m_sptr_resource{other.m_sptr_resource}
    {
    }

    /* Destructor */

    ~pmimallocator() {}

    /* Allocate */

    [[nodiscard]] T* allocate(const std::size_t n /*, const std::size_t alignment = 0 */)
    {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
        {
            throw std::bad_alloc();    // Check for overflow
        }
        // fmt::print("Resource use count : {} \n", m_sptr_resource.use_count());
        return static_cast<pointer>(m_sptr_resource->allocate(n * sizeof(T) /*, alignment*/));
    }

    /* Deallocate */

    void deallocate(T* p, std::size_t n = 0)
    {
        void* tmp = static_cast<void*>(p);
        return m_sptr_resource->deallocate(tmp, n);
    }

    /* Max size */

    size_type max_size() const noexcept
    {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    // /* Usable size (only with mimalloc) */
    // #if WITH_MIMALLOC
    //     std::size_t get_usable_size() { return m_sptr_resource->get_usable_size(); }
    // #endif

    /* Construct */

    // void construct( pointer p, const_reference val );

    // template <class U, class... Args>
    // void construct(U* p, Args&&... args) {
    //     ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    // }

    /* Destroy */

    // void destroy( pointer p );

    // template <class U>
    // void destroy(U* p) {
    //     p->~U();
    // }

    friend bool operator==(const pmimallocator& lhs, const pmimallocator& rhs)
    {
        return (lhs.m_sptr_resource == rhs.m_sptr_resource);
    }

    friend bool operator!=(const pmimallocator& lhs, const pmimallocator& rhs)
    {
        return (lhs.m_sptr_resource != rhs.m_sptr_resource);
    }

private:
    template <typename U, typename R>
    friend class pmimallocator;

    shared_resource m_sptr_resource;
};

// TODO (optional) : All of this

/* Address */

// pointer address( reference x ) const;

// pointer address( reference x ) const noexcept;

// const_pointer address( const_reference x ) const;

// const_pointer address( const_reference x ) const noexcept;

/* Allocate at least */

// [[nodiscard]] constexpr std::allocation_result<T*, std::size_t>
// allocate_at_least( std::size_t n );
