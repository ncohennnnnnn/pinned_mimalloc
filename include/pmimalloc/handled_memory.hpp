#pragma once

template <typename Base>
/** @brief Memory ready to be handled. */
class handled_memory : public Base
{
public:
    handled_memory() = default;

    handled_memory(void* ptr, const std::size_t size)
      : m_address{ptr}
      , m_size{size}
    {
    }

    void* get_address(void)
    {
        return m_address;
    }

    std::size_t get_size(void)
    {
        return m_size;
    }

protected:
    void* m_address;
    std::size_t m_size;
};
