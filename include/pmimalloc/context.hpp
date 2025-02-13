#include <cstdlib>
#include <utility>
//
#include <fmt/core.h>
//
#include <pmimalloc/concepts.hpp>

class backend_none;

template <typename Memory, typename Backend = backend_none>
/** @brief Manages and brings together the choice of the registration (Backend)
 * and pinning mechanism (Memory), and memory type to use (Memory).
*/
class context : public Memory
{
public:
    using memory_t = Memory;
    using backend_t = Backend;
    using rkey_t = typename backend_t::rkey_t;

    struct key_t
    {
        rkey_t remote_key;
        std::uint64_t offset;
        std::uint64_t get()
        {
            return (reinterpret_cast<std::uint64_t>(remote_key) + offset);
        }
    };

    context()
      : memory_t{}
      , m_backend{}
      , m_rkey{}
    {
    }

    template <typename... Args>
    context(Args&&... args)
      : memory_t{std::forward<Args>(args)...}
      , m_backend{memory_t::get_address(), memory_t::get_size()}
      , m_rkey{m_backend.get_remote_key()}
    {
    }

    context(const context& other) = delete;

    ~context() {}

    template <typename T>
    key_t get_key(T* ptr)
    {
        std::ptrdiff_t diff =
            reinterpret_cast<uintptr_t>(memory_t::get_address()) - reinterpret_cast<uintptr_t>(ptr);
        if (diff < 0)
        {
            fmt::print("{} : [error] Pointer not in arena.", reinterpret_cast<uintptr_t>(ptr));
            return NULL;
        }
        return {m_rkey, reinterpret_cast<uint32_t>(diff)};
    }

    rkey_t get_key()
    {
        return m_rkey;
    }

protected:
    backend_t m_backend;
    rkey_t m_rkey;
};

/*------------------------------------------------------------------*/
/*                        Default backend                           */
/*------------------------------------------------------------------*/

class backend_none
{
public:
    using rkey_t = uint8_t;

    backend_none() {}

    backend_none(backend_none&& /*other*/) noexcept {}

    backend_none(void* /*ptr*/, const std::size_t /*size*/) {}

    backend_none& operator=(backend_none&& /*other*/) noexcept
    {
        return *this;
    }

    ~backend_none() {}

    int deregister(void) const
    {
        return 0;
    }

    template <typename... Args>
    static inline int register_memory(Args&&... args)
    {
        return 0;
    }

    static inline int register_memory(void* /*ptr*/, std::size_t /*base_size*/)
    {
        return 0;
    }

    template <typename T>
    static inline int register_ptr(T* ptr, void* base_ptr, std::size_t base_size)
    {
        return 0;
    }

    static inline void* get_local_key()
    {
        return nullptr;
    }

    static inline rkey_t get_remote_key()
    {
        return 0;
    }

    static inline rkey_t get_remote_key(void* /*ptr*/)
    {
        return 0;
    }
};
