class base
{
protected:
    void set_address(void* ptr)
    {
        m_address = ptr;
    }
    void set_size(std::size_t s)
    {
        m_size = s;
    }
    void set_numa_node(int n)
    {
        m_numa_node = n;
    }
    void* m_address;
    std::size_t m_size;
    int m_numa_node = -1;
};