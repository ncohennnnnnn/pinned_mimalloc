#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <vector>
#include <memory>

#include <resource.hpp>

 
template<typename T, typename Resource>
class pmimallocator
{
public:
/* Types */
    using this_type       = pmimallocator<T, Resource>;
    using resource_type   = Resource;
    using shared_resource = std::shared_ptr<Resource>;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;

    template<typename U>
    struct rebind { 
        using other = pmimallocator<U,Resource>; 
    };

/* Contructors */

    pmimallocator( shared_resource r) noexcept 
    : m_sptr_resource{r} 
    {}

    pmimallocator( const resource_builder<resource_type>& rb ) noexcept 
    : m_sptr_resource{rb.sbuild()} 
    {}

    pmimallocator( const resource_type& r ) noexcept 
    // : m_sptr_resource{std::make_shared<rb::resource_t>(rb.build())} 
    {
        m_sptr_resource = std::make_shared<resource_type>(r);
    }

    pmimallocator( const pmimallocator& other ) noexcept = delete;

// TODO : take care of this one
    template< class U >
    pmimallocator( const pmimallocator<Resource, U>& other ) noexcept = delete;

/* Destructor */

    ~pmimallocator() {}

/* Allocate */

    [[nodiscard]] constexpr T* allocate( const std::size_t n, const std::size_t alignment = 0 )    
    {
        void* rtn = m_sptr_resource->allocate(n, alignment);
        return static_cast<T*>(rtn);
    }

/* Deallocate */

    void deallocate( T* p, std::size_t n = 0 )
    {
        void* tmp = static_cast<void*>(p);
        return m_sptr_resource->deallocate(tmp, n);
    }

		
private:
    shared_resource m_sptr_resource;
};
 

/* Operators */

    template< class Resource1, class Resource2, class T1, class T2 >
    bool operator==( const pmimallocator<Resource1, T1>& lhs, const pmimallocator<Resource2, T2>& rhs ) = delete;

    template< class Resource1, class Resource2, class T1, class T2 >
    constexpr bool operator==( const pmimallocator<Resource2, T1>& lhs, const pmimallocator<Resource2, T2>& rhs ) = delete;

    template< class Resource1, class Resource2, class T1, class T2 >
    bool operator!=( const pmimallocator<Resource1, T1>& lhs, const pmimallocator<Resource2, T2>& rhs ) = delete;







// TODO : All of this (optional)

/* Address */

    // pointer address( reference x ) const;

    // pointer address( reference x ) const noexcept;

    // const_pointer address( const_reference x ) const;

    // const_pointer address( const_reference x ) const noexcept;

/* Allocate at least */

    // [[nodiscard]] constexpr std::allocation_result<T*, std::size_t> 
    // allocate_at_least( std::size_t n );

/* Max size */

    // size_type max_size() const noexcept;

/* Construct */

    // void construct( pointer p, const_reference val );

    // template< class U, class... Args >
    // void construct( U* p, Args&&... args );

/* Destroy */

    // void destroy( pointer p );

    // template<class U>
    // void destroy( U* p );