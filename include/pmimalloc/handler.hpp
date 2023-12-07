#pragma once

template <typename Context>
class handler : public Context
{
public:
    using context_t = Context;
    using this_type = handler<context_t>;
    using shared_this = std::shared_ptr<handler<context_t>>;
    using key_t = context_t::key_t;

    handler() = default;

    handler(context_t&& c) noexcept
      : context_t{std::move(c)}
    {
    }

    handler(const this_type& r) noexcept = delete;

    handler(void* ptr, const std::size_t size)
      : context_t{ptr, size}
    {
    }

    template <typename T>
    key_t get_key(T* ptr)
    {
        return this->template get_key(ptr);
    }

    auto get_key(void)
    {
        return this->template get_key();
    }
};