#include <backend.hpp>


int backend::deregister(void) {
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

template<typename T>
inline int backend::register_ptr(T* ptr, void* base_ptr, std::size_t base_size) const {
    void* ptr_tmp = static_cast<void*>(ptr);
    uint64_t offset;
    offset = (uint64_t)((char*)ptr_tmp - (char*)base_ptr);
    return fi_mr_reg(m_domain, base_ptr, base_size, flags(), offset, 0, 0, &m_region, NULL); 
}

int backend::_build_base() {
    int ret;
    ret = _build_info();
    if (ret) { perror("fi_getinfo"); return ret;}
    ret = _build_fabric();
    if (ret) { perror("fi_getfabric"); return ret;}
    ret = _build_domain();
    if (ret) { perror("fi_getdomain"); return ret;}
}