class backend_none
{
public:
    using key_t = uint8_t;

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

    static inline key_t get_remote_key()
    {
        return 0;
    }

    static inline key_t get_remote_key(void* /*ptr*/)
    {
        return 0;
    }
};
