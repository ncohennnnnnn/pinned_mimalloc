#pragma once

#include <pmimalloc.hpp>

/** @brief context manages and brings together the choice of the backend,
 * pinning mechanism, and memory type to use.
 * @tparam memory_type_alloc to choose if allocate on device or host
 * @tparam memory_type_pin to choose if mlock or cudaHostRegister pins the memory
 * 
 * TODO: Throw in some concepts to make sure one doesn't cudaMalloc + pin.
*/
template<typename backend, typename memory_type_alloc, typename memory_type_pin = memory_type_alloc>
class context {
    using key_t = backend::key_t;

public:
    context(std::size_t size, bool pin = true, bool alignement = 0) : m_memory{}, m_backend{}
    {
        m_numa_node = m_memory.allocate(size, alignement);
        if (pin) { pin(); }
        m_backend.register_memory(m_memory.get_address(), m_memory.get_size());
    }

    context(const context& other) = delete;

    ~context() 
    { 
        void* ptr = m_memory.get_address();
        std::size_t size = m_memory.get_size();
        ~m_backend(); 
        if (m_pinned) { unpin(); }
        deallocate(); 
    }

    void pin() { memory_type_pin::pin_or_unpin(m_memory.get_address(), m_memory.get_size(), true); }

    void unpin() { memory_type_pin::pin_or_unpin(m_memory.get_address(), m_memory.get_size(), false); }

    // void* allocate_aligned(std::size_t size) { return memory_type_alloc::allocated_aligned(size); }

    void deallocate() { return memory_type_alloc::deallocate(); }

    std::size_t get_size(void) { return m_memory.get_size(); }

    void* get_address(void) { return m_memory.get_address(); }

    int get_numa_node(void) { return m_numa_node; }

    template<typename T>
    key_t get_key(T* ptr) { 
        void* ptr_tmp = static_cast<void*>(ptr);
        return m_backend.get_remote_key(ptr_tmp); 
    }

private:
    bool m_pinned;
    std::size_t m_numa_node;
    memory_type_alloc m_memory;
    backend m_backend;
};