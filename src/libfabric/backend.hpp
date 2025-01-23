#include <utility>
//
#include <fmt/core.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

class backend
{
public:
    /* The internal memory region handle */
    using info_t = struct fi_info;
    using fabric_t = struct fid_fabric;
    using region_t = struct fid_mr;
    using domain_t = struct fid_domain;

    using rkey_t = uint64_t;

    /* Default constructor creates unusable handle (region) */
    backend()
      : m_region{}
    {
        _build_base();
    }

    template <typename Memory>
    backend(const Memory& c) noexcept
      : m_region{}
    {
        _build_base();
        register_memory(c.get_address(), c.get_size());
    }

    backend(void* ptr, const std::size_t size) noexcept
      : m_region{}
    {
        _build_base();
        register_memory(ptr, size);
    }

    /* Move constructor, clear other region so that it is not unregistered twice */
    backend(backend&& other) noexcept
      : m_region{std::exchange(other.m_region, nullptr)}
      , m_domain{std::exchange(other.m_domain, nullptr)}
      , m_fabric{std::exchange(other.m_fabric, nullptr)}
      , m_info{std::exchange(other.m_info, nullptr)}
    {
    }

    /* Move assignment, clear other region so that it is not unregistered twice */
    backend& operator=(backend&& other) noexcept
    {
        m_region = std::exchange(other.m_region, nullptr);
        m_domain = std::exchange(other.m_domain, nullptr);
        m_fabric = std::exchange(other.m_fabric, nullptr);
        m_info = std::exchange(other.m_info, nullptr);
        return *this;
    }

    ~backend() noexcept
    {
        deregister();
        fi_close(&(m_domain->fid));
        fi_close(&(m_fabric->fid));
        fi_freeinfo(m_info);
    }

    /* Deregister the memory region. Returns 0 when successful, -1 otherwise */
    int deregister(void);

    inline int register_memory(void* ptr, std::size_t base_size)
    {
        return fi_mr_reg(m_domain, ptr, base_size, flags(), 0, 0, 0, &m_region, NULL);
    }

    template <typename T>
    inline int register_ptr(T* ptr, void* base_ptr, std::size_t base_size) /* const */;

    /* Default registration flags for this provider */
    static inline constexpr int flags()
    {
        return FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
    }

    void release_region() noexcept
    {
        m_region = nullptr;
    }

    /* Get the local descriptor of the memory region. */
    inline void* get_local_descriptor()
    {
        return fi_mr_desc(m_region);
    }

    /* Get the remote key of the memory region. */
    inline rkey_t get_remote_key()
    {
        return fi_mr_key(m_region);
    }

    /* Returns the underlying libfabric region handle */
    inline region_t get_region() const
    {
        return *m_region;
    }

    info_t get_info()
    {
        return *m_info;
    }

    fabric_t get_fabric()
    {
        return *m_fabric;
    }

    domain_t get_domain()
    {
        return *m_domain;
    }

private:
    int _build_info()
    {
        info_t* hints = fi_allocinfo();
        return fi_getinfo(FI_MAJOR_VERSION, NULL, NULL, 0, hints, &m_info);
    }

    int _build_fabric()
    {
        return fi_fabric(m_info->fabric_attr, &m_fabric, NULL);
    }

    int _build_domain()
    {
        return fi_domain(m_fabric, m_info, &m_domain, NULL);
    }

    int _build_base();

protected:
    region_t* m_region;
    domain_t* m_domain;
    fabric_t* m_fabric;
    info_t* m_info;
};
