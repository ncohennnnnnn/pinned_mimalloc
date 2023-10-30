class backend_none {

    using key_t    = uint8_t;

    backend_none() {}

    backend_none(backend_none&& other) noexcept {}

    backend_none& operator=(backend_none&& other) noexcept {}

    ~backend_none() noexcept {}

    int deregister(void) const { return 0; }

    template<typename... Args>
    static inline int register_memory(Args&&... args) { return 0; }

    static inline int register_memory(void* ptr, std::size_t base_size) const { return 0; }

    template<typename T>
    static inline int register_ptr(T* ptr, void* base_ptr, std::size_t base_size) const { return 0; }

    static inline void* get_local_key() { return 0; }

    static inline key_t get_remote_key() { return 0; }

    static inline key_t get_remote_key(void* ptr) { return 0; }

};
