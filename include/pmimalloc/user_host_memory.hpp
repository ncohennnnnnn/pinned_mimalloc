template <typename Base>
/** @brief Already allocated memory living on the host. */
class user_host_memory : public Base
{
public:
    user_host_memory()
      : Base{}
      , m_address{nullptr}
      , m_size{0}
      , m_numa_node{-1}
    {
    }

    user_host_memory(void* ptr, const std::size_t size)
      : Base{}
      , m_address{ptr}
      , m_size{size}
      , m_numa_node{-1}
    {
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
    }

    ~user_host_memory() {}

    void* get_address(void)
    {
        return m_address;
    }

    std::size_t get_size(void)
    {
        return m_size;
    }

    int get_numa_node(void)
    {
        return m_numa_node;
    }

protected:
    void* m_address;
    std::size_t m_size;
    int m_numa_node;
};
