#pragma once

#include <register.hpp>


// default implementation: call normal registration
template<class context>
constexpr auto
register_device_memory(context&& c, void* ptr, std::size_t size) noexcept(
    noexcept(register_memory(std::forward<context>(c), ptr, size)))
    -> decltype(register_memory(std::forward<context>(c), ptr, size))
{
    return register_memory(std::forward<context>(c), ptr, size);
}

struct register_device_fn
{
    template<typename context>
    constexpr auto operator()(context&& c, void* ptr, std::size_t size) const
        noexcept(noexcept(register_device_memory(std::forward<context>(c), ptr, size)))
            -> decltype(register_device_memory(std::forward<context>(c), ptr, size))
    {
        return register_device_memory(std::forward<context>(c), ptr, size);
    }
};

constexpr auto const& register_device_memory = static_const_v<register_device_fn>;
