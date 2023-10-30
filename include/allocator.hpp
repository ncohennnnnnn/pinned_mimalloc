#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

#include <resource.hpp>

 
template<typename T, typename Resource>
class pmimallocator
{
/* Types */
    using this_type       = pmimallocator<T, Resource>;
    using resource_type   = Resource;
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
    struct rebind { typedef pmimallocator<U> other; };

/* Contructors */

    pmimallocator() throw();

    pmimallocator() noexcept;

    constexpr pmimallocator() noexcept;

    pmimallocator( const pmimallocator& other ) throw();

    pmimallocator( const pmimallocator& other ) noexcept;

    constexpr pmimallocator( const pmimallocator& other ) noexcept;

    template< class U >
    pmimallocator( const pmimallocator<U>& other ) throw();

    template< class U >
    pmimallocator( const pmimallocator<U>& other ) noexcept;

    template< class U >
    constexpr pmimallocator( const pmimallocator<U>& other ) noexcept;
 
    pmimallocator() = default;

/* Destructors */

    ~allocator();

    constexpr ~allocator();

/* Address */

    pointer address( reference x ) const;

    pointer address( reference x ) const noexcept;

    const_pointer address( const_reference x ) const;

    const_pointer address( const_reference x ) const noexcept;

/* Allocate */

    pointer allocate( size_type n, const void* hint = 0 );

    T* allocate( std::size_t n, const void* hint );

    T* allocate( std::size_t n );

    [[nodiscard]] constexpr T* allocate( std::size_t n )    
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
 
        if (auto p = static_cast<T*>(std::malloc(size * sizeof(T))))
        {
            report(p, n);
            return p;
        }
 
        throw std::bad_alloc();
    }
    


/* Allocate at least */

    [[nodiscard]] constexpr std::allocation_result<T*, std::size_t> 
    allocate_at_least( std::size_t n );

/* Deallocate */

    void deallocate( T* p, std::size_t n );

    constexpr void deallocate( T* p, std::size_t n );

/* Max size */

    size_type max_size() const throw();

    size_type max_size() const noexcept;

/* Construct */

    void construct( pointer p, const_reference val );

    template< class U, class... Args >
    void construct( U* p, Args&&... args );

/* Destroy */

    void destroy( pointer p );

    template<class U>
    void destroy( U* p );
		



    
    void deallocate(T* p, std::size_t n) noexcept
    {
        report(p, n, 0);
        std::free(p);
    }
private:
    void report(T* p, std::size_t n, bool alloc = true) const
    {
        std::cout << (alloc ? "Alloc: " : "Dealloc: ") << sizeof(T) * n
                  << " bytes at " << std::hex << std::showbase
                  << reinterpret_cast<void*>(p) << std::dec << '\n';
    }
};
 

/* Operators */

    template< class T1, class T2 >
    bool operator==( const pmimallocator<T1>& lhs, const pmimallocator<T2>& rhs ) throw();

    template< class T1, class T2 >
    bool operator==( const pmimallocator<T1>& lhs, const pmimallocator<T2>& rhs ) noexcept;

    template< class T1, class T2 >
    constexpr bool
        operator==( const pmimallocator<T1>& lhs, const pmimallocator<T2>& rhs ) noexcept;

    template< class T1, class T2 >
    bool operator!=( const pmimallocator<T1>& lhs, const pmimallocator<T2>& rhs ) throw();

    template< class T1, class T2 >
    bool operator!=( const pmimallocator<T1>& lhs, const pmimallocator<T2>& rhs ) noexcept;