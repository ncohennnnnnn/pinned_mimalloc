#pragma once

#include <register.hpp>


// default implementation: call normal registration
template<class Context>
constexpr auto
register_device_memory(Context&& c, void* ptr, std::size_t size) noexcept(
    noexcept(register_memory(std::forward<Context>(c), ptr, size)))
    -> decltype(register_memory(std::forward<Context>(c), ptr, size))
{
    return register_memory(std::forward<Context>(c), ptr, size);
}

struct register_device_fn
{
    template<typename Context>
    constexpr auto operator()(Context&& c, void* ptr, std::size_t size) const
        noexcept(noexcept(register_device_memory(std::forward<Context>(c), ptr, size)))
            -> decltype(register_device_memory(std::forward<Context>(c), ptr, size))
    {
        return register_device_memory(std::forward<Context>(c), ptr, size);
    }
};

constexpr auto const& register_device_memory = static_const_v<register_device_fn>;
