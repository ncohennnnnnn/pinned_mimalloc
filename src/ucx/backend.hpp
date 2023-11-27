#include <fmt>

class backend
{
public:
    // The internal memory region handle
    using info_t = struct fi_info;
    using fabric_t = struct fid_fabric;
    using region_t = struct fid_mr;
    using domain_t = struct fid_domain;

    using key_t = uint64_t;

    // Default constructor creates unusable handle (region)
    backend()
      : m_region{nullptr}
    {
        _build_base();
        return *this;
    }

    // backend(backend const&) noexcept = default;
    // backend& operator=(backend const&) noexcept = default;

    // backend(region_t* region, unsigned char* addr,
    //     /*std::size_t size , uint32_t flags*/) noexcept
    // :
    // // m_address{addr}
    // , m_region{region}
    // // , m_size{uint32_t(size)}
    // {
    //     _build_base();
    //     return *this;
    //     //            DEBUG(NS_MEMORY::mrn_deb,
    //     //                trace(NS_DEBUG::str<>("backend"), *this));
    // }

    // move constructor, clear other region so that it is not unregistered twice
    backend(backend&& other) noexcept
      :
      , m_region{std::exchange(other.m_region, nullptr)}
      , m_info{std::exchange(other.m_info, nullptr)}
      , m_domain{std::exchange(other.m_domain, nullptr)}
      , m_fabric{std::exchange(other.m_fabric, nullptr)}
    {
    }

    // move assignment, clear other region so that it is not unregistered twice
    backend& operator=(backend&& other) noexcept
    {
        m_region = std::exchange(other.m_region, nullptr);
        m_info = std::exchange(other.m_info, nullptr);
        m_domain = std::exchange(other.m_domain, nullptr);
        m_fabric = std::exchange(other.m_fabric, nullptr);
        return *this;
    }

    ~backend() noexcept
    {
        deregister();
        fi_close(&m_domain->fid);
        fi_close(&m_fabric->fid);
        fi_freeinfo(m_info);
    }

    // --------------------------------------------------------------------
    // Deregister the memory region. Returns 0 when successful, -1 otherwise
    int deregister(void) const
    {
        if (m_region /*&& !get_user_region()*/)
        {
            // DEBUG(NS_MEMORY::mrn_deb, trace(NS_DEBUG::str<>("release"), m_region));
            //
            if (fi_close(&m_region->fid))
            {
                // DEBUG(NS_MEMORY::mrn_deb, error("fi_close mr failed"));
                fmt::print("fi_close mr failed");
                return -1;
            }
            // else
            // {
            //     DEBUG(NS_MEMORY::mrn_deb, trace(NS_DEBUG::str<>("de-Registered region"), *this));
            // }
            m_region = nullptr;
        }
        return 0;
    }

    // register region
    template <typename... Args>
    static inline int register_memory(Args&&... args)
    {
        return fi_mr_reg(std::forward<Args>(args)...);
    }

    // static inline int register_memory(void) const {
    //     return fi_mr_reg(m_domain, m_address, m_size, flags(), 0, 0, 0, &m_region, NULL);
    // }

    static inline int register_memory(void* ptr, std::size_t base_size) const
    {
        return fi_mr_reg(m_domain, ptr, base_size, flags(), 0, 0, 0, &m_region, NULL);
    }

    // template<typename T>
    // static inline int register(T* ptr) const {
    //     void* ptr_tmp = static_cast<void*>(ptr);
    //     uint64_t offset;
    //     offset = (uint64_t)((char*)ptr_tmp - (char*)m_address);
    //     return fi_mr_reg(m_domain, m_address, m_size, flags(), offset, 0, 0, &m_region, NULL);
    // }

    template <typename T>
    static inline int register_ptr(T* ptr, void* base_ptr, std::size_t base_size) const
    {
        void* ptr_tmp = static_cast<void*>(ptr);
        uint64_t offset;
        offset = (uint64_t) ((char*) ptr_tmp - (char*) base_ptr);
        return fi_mr_reg(m_domain, base_ptr, base_size, flags(), offset, 0, 0, &m_region, NULL);
    }

    // // register region
    // template<typename... Args>
    // static inline int register_memory_attr(Args&&... args)
    // {
    //     // [[maybe_unused]] auto scp = NS_MEMORY::mrn_deb.scope(__func__, std::forward<Args>(args)...);
    //     //        int x = FI_HMEM_ROCR;
    //     //        fi_mr_regattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
    //     //                    uint64_t flags, struct fid_mr **mr)
    //     return fi_mr_regattr(std::forward<Args>(args)...);
    // }

    // Default registration flags for this provider
    static inline constexpr int flags()
    {
        return FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
    }

    // Get the local descriptor of the memory region.
    static inline void* get_local_key()
    {
        return fi_mr_desc(m_region);
    }

    // Get the remote key of the memory region.
    static inline key_t get_remote_key()
    {
        return fi_mr_key(m_region);
    }

    // Get the remote key of an object inside the memory region.
    static inline key_t get_remote_key(void* ptr)
    {
        return fi_mr_key(m_region);
    }

    // // Return the address of this memory region block.
    // inline unsigned char* get_address(void) const { return m_address; }

    // Get the size of the memory chunk usable by this memory region,
    // this may be smaller than the value returned by get_length
    // if the region is a sub region (partial region) within another block
    // inline uint64_t get_size(void) const { return m_size; }

    // // Get the size used by a message in the memory region.
    // inline uint32_t get_message_length(void) const { return m_used_space; }

    // // Set the size used by a message in the memory region.
    // inline void set_message_length(uint32_t length) { m_used_space = length; }

    void release_region() noexcept
    {
        m_region = nullptr;
    }

    // return the underlying libfabric region handle
    inline region_t* get_region() const
    {
        return m_region;
    }

    // --------------------------------------------------------------------
    //     friend std::ostream& operator<<(std::ostream& os, backend const& region)
    //     {
    //         (void)region;
    // #if has_debug
    //         // clang-format off
    //             os /*<< "region "*/      << NS_DEBUG::ptr(&region)
    //                //<< " fi_region "  << NS_DEBUG::ptr(region.m_region)
    //            << " address "    << NS_DEBUG::ptr(region.m_address)
    //            << " size "       << NS_DEBUG::hex<6>(region.m_size)
    //                //<< " used_space " << NS_DEBUG::hex<6>(region.m_used_space/*m_size*/)
    //                << " loc key "  << NS_DEBUG::ptr(region.m_region ? region_provider::get_local_key(region.m_region) : nullptr)
    //                << " rem key " << NS_DEBUG::ptr(region.m_region ? region_provider::get_remote_key(region.m_region) : 0);
    //         // clang-format on
    // #endif
    //         return os;
    //     }

    info_t get_info()
    {
        return m_info;
    }

    fabric_t get_fabric()
    {
        return m_fabric;
    }

    domain_t get_domain()
    {
        return m_domain;
    }

private:
    // TODO : Maybe adapt the parameters
    int _build_info()
    {
        info_t hints = fi_allocinfo();
        return fi_getinfo(FIVER, , /*"1234"*/, /*flags*/ 0, hints, &m_info);
    }

    // TODO : Maybe put smth instead of NULL
    int _build_fabric()
    {
        return fi_fabric(m_info->fabric_attr, &m_fabric, NULL);
    }

    // TODO : Maybe put smth instead of NULL
    int _build_domain()
    {
        return fi_domain(m_fabric, m_info, &m_domain, NULL);
    }

    int _build_base()
    {
        int ret;
        ret = _build_info();
        if (ret)
        {
            perror("fi_getinfo");
            return ret;
        }
        ret = _build_fabric();
        if (ret)
        {
            perror("fi_getfabric");
            return ret;
        }
        ret = _build_domain();
        if (ret)
        {
            perror("fi_getdomain");
            return ret;
        }
    }

protected:
    // This gives the start address of this region.
    // This is the address that should be used for data storage
    // unsigned char* m_address;

    // The hardware level handle to the region (as returned from libfabric fi_mr_reg)
    mutable region_t* m_region;

    // The (maximum available) size of the memory buffer
    // uint32_t m_size;

    // Space used by a message in the memory region.
    // This may be smaller/less than the size available if more space
    // was allocated than it turns out was needed
    // mutable uint32_t m_used_space;

    domain_t m_domain;
    fabric_t m_fabric;
    info_t m_info;
};
