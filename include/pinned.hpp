#pragma once

#include <sys/mman.h>

#include <fmt/core.h>

template <typename Memory>
/** @brief Pinned memory living on the host. */
class pinned : public Memory
{
public:
    pinned()
      : Memory{}
      , m_pinned{false}
    {
    }

    pinned(Memory&& mem)
      : Memory{std::move(mem)}
      , m_pinned{false}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(const std::size_t size, const std::size_t alignement = 0)
      : Memory{size, alignement}
      , m_pinned{false}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    pinned(void* ptr, const std::size_t size)
      : Memory{ptr, size}
      , m_pinned{false}
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

protected:
    bool m_pinned;
};