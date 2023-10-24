#pragma once

#include <cstddef>
#include <utility>


// Customization point for memory registration
// ===========================================
// The function
//
//     template<memory_type M, typename Context>
//     /*unspecified*/ register_memory<M>(Context&& context, void* ptr, std::size_t size)
//
// is found by ADL and can be used to customize memory registration for different network/transport
// layers. The memory at address `ptr' and extent `size' shall be registered with the `context'.
// The function is mandated to return a memory region object managing the lifetime of the
// registration (i.e. when destroyed, deregistration is supposed to happen). This class R must
// additionally satisfy the following requirements:
//
// - MoveConstructible
// - MoveAssignable
//
// Given
// - r:   object of R
// - cr:  const object of R
// - h:   object of R::handle_type
// - o,s: values convertable to std::size_t
//
// inner types:
// +------------------+---------------------------------------------------------------------------+
// | type-id          | requirements                                                              |
// +------------------+---------------------------------------------------------------------------+
// | R::handle_type   | satisfies DefaultConstructible, CopyConstructible, CopyAssignable         |
// +------------------+---------------------------------------------------------------------------+
//
// operations on R:
// +---------------------+----------------+-------------------------------------------------------+
// | expression          | return type    | requirements                                          |
// +---------------------+----------------+-------------------------------------------------------+
// | cr.get_handle(o, s) | R::handle_type | returns RMA handle at offset o from base address ptr  |
// |                     |                | with size s                                           |
// +---------------------+----------------+-------------------------------------------------------+
// | r.~R()              |                | deregisters memory if not moved-from                  |
// +---------------------+----------------+-------------------------------------------------------+
//
// operations on handle_type:
// +---------------------+----------------+-------------------------------------------------------+
// | expression          | return type    | requirements                                          |
// +---------------------+----------------+-------------------------------------------------------+
// | h.get_local_key()   | unspecified    | returns local rma key                                 |
// +---------------------+----------------+-------------------------------------------------------+
// | h.get_remote_key()  | unspecified    | returns rma key for remote access                     |
// +---------------------+----------------+-------------------------------------------------------+
//

struct register_fn
{
    template<typename Context>
    constexpr auto operator()(Context&& c, void* ptr, std::size_t size) const
        noexcept(noexcept(register_memory(std::forward<Context>(c), ptr, size)))
            -> decltype(register_memory(std::forward<Context>(c), ptr, size))
    {
        return register_memory(std::forward<Context>(c), ptr, size);
    }
};

template<class T>
constexpr T static_const_v{};

constexpr auto const& register_memory = static_const_v<register_fn>;
