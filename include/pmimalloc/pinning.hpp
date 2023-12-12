#pragma once

#include <sys/mman.h>

#include <fmt/core.h>

/*------------------------------------------------------------------*/
/*                            No pinning                            */
/*------------------------------------------------------------------*/

template <class Memory>
/** @brief Host or device memory that is not to be pinned. */
class not_pinned : public Memory
{
public:
    not_pinned()
      : Memory{}
    {
    }

    not_pinned(Memory&& mem)
      : Memory{std::move(mem)}
    {
    }

    not_pinned(const std::size_t size, const std::size_t alignment = 0)
      : Memory{size, alignment}
    {
    }

    not_pinned(void* ptr, const std::size_t size)
      : Memory{ptr, size}
    {
    }

    not_pinned(void* ptr_a, void* ptr_b, const std::size_t size)
      : Memory{ptr_a, ptr_b, size}
    {
    }

    not_pinned(not_pinned&&) noexcept = default;

    ~not_pinned() {}

private:
    bool m_pinned = false;
};

/*------------------------------------------------------------------*/
/*                         mlock pinning                            */
/*------------------------------------------------------------------*/

template <typename Memory>
/** @brief Pinned memory living on the host. */
class pinned : public Memory
{
public:
    pinned()
      : Memory{}
    {
    }

    pinned(Memory&& mem)
      : Memory{std::move(mem)}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(const std::size_t size, const std::size_t alignment = 0)
      : Memory{size, alignment}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(void* ptr, const std::size_t size)
      : Memory{ptr, size}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(void* ptr_a, void* ptr_b, const std::size_t size)
      : Memory{ptr_a, ptr_b, size}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(pinned&&) noexcept = default;

    ~pinned()
    {
        if (m_pinned)
        {
            _pin_or_unpin(Memory::m_address, Memory::m_size, false);
        }
    }

private:
    bool _pin_or_unpin(void* ptr, const size_t size, const bool pin)
    {
        int success;
        std::string str;
        if (pin)
        {
            success = mlock(ptr, size);
            m_pinned = true;
            str = "pin";
        }
        else
        {
            success = munlock(ptr, size);
            m_pinned = false;
            str = "unpin";
        }
        if (success != 0)
        {
            fmt::print("{} : [error] failed to {} the allocated memory : ", ptr, str);
            if (errno == EAGAIN)
            {
                fmt::print("EAGAIN. \n (Some or all of the specified address range could not be "
                           "locked.) \n");
            }
            else if (errno == EINVAL)
            {
                fmt::print("EINVAL. \n (The result of the addition addr+len was less than addr. "
                           "addr = {} and len = {})\n",
                    ptr, size);
            }
            else if (errno == ENOMEM)
            {
                fmt::print("ENOMEM. \n (Some of the specified address range does not correspond to "
                           "mapped pages in the address space of the process.) \n");
            }
            else if (errno == EPERM)
            {
                fmt::print("EPERM. \n (The caller was not privileged.) \n");
            }
        }
        else
        {
            fmt::print("{} : Memory {}ned. \n", ptr, str);
        }
        return (bool) (1 - success);
    }

    bool m_pinned = false;
};