#pragma once

#include <concepts>

template <typename T>
concept key = std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>;

template <typename T>
concept Backend = requires(T& r, void* ptr, void* ptr_, std::size_t s)
{
    // requires std::movable<T>;
    {
        r.get_key(ptr)
        } -> key;
    {
        r.register_memory(ptr, s)
        } -> std::integral;
    {
        r.register_ptr(ptr_, ptr, s)
        } -> std::integral;
};
