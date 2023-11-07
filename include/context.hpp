#include <cstdlib>
#include <utility>

// class backend {
// public:
//     int deregister(void);

//     static inline int register_memory(void* ptr, std::size_t base_size);

//     template<typename T>
//     static inline int register_ptr(T* ptr, void* base_ptr, std::size_t base_size);
// };

template<typename Memory, typename Backend>
/** @brief Manages and brings together the choice of the registration (Backend)
 * and pinning mechanism (Memory), and memory type to use (Memory).
 * 
 * TODO: Throw in some concepts.
*/
class context : public Memory {
public:
    using memory_t  = Memory;
    using backend_t = Backend;
    using key_t     = typename Backend::key_t;

    context()
    : memory_t{}
    , m_backend{}
    {}

    context(memory_t&& mem) 
    : memory_t{std::move(mem)}
    , m_backend{&mem} 
    {}

    context(const std::size_t size, const std::size_t alignement = 0) 
    : memory_t{size, alignement}
    , m_backend{Memory::m_address, Memory::m_size}
    {}

    context(void* ptr, const std::size_t size) 
    : memory_t{ptr, size}
    , m_backend{Memory::m_address, Memory::m_size}
    {}

    context(const context& other) = delete;

    ~context() {}

    template<typename T>
    key_t get_key(T* ptr) 
    { 
        void* ptr_tmp = static_cast<void*>(ptr);
        return m_backend.get_remote_key(ptr_tmp); 
    }

protected:
    backend_t m_backend;
};