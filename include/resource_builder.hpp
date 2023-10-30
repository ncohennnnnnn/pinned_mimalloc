#pragma once 

#include <resource.hpp>

class resource_builder {
public:

    resource_builder() noexcept
    : m_on_device{false}
    , m_pin_with_device{false}
    , m_register{false}
    , m_alignement{0}
    , m_size{0}
    {}

    ~resource_builder() noexcept {}

    resource_builder on_device() 
    { 
        m_on_device = true;
        return *this;
    }

    resource_builder pin() 
    { 
        m_pin = true; 
        return *this;
    }

    resource_builder pin_with_device() 
    { 
        m_pin_with_device = true; 
        return *this;
    }

    resource_builder register() 
    { 
        m_register = true; 
        return *this;
    }

    resource_builder set_size(std::size_t size) 
    { 
        m_size = size; 
        return *this;
    }

    resource_builder alignement(std::size_t size) 
    { 
        m_alignement = size; 
        return *this;
    }

    resource build() {
        if (m_register && m_on_device)
        { 
            resource<context<backend, device_memory>> rtn(m_size, false, alignement);
            reset();
            return rtn; 
        }

        if (m_register && !m_on_device && m_pin && m_pin_with_device)
        { 
            resource<context<backend, host_memory, device_memory>> rtn(m_size, true, alignement);
            reset();
            return rtn; 
        }

        if ((m_register && !m_on_device && m_pin && !m_pin_with_device) || (m_register && !m_on_device && !m_pin))
        { 
            resource<context<backend, host_memory>> rtn(m_size, m_pin, alignement);
            reset();
            return rtn; 
        }

        if (!m_register && m_on_device)
        { 
            resource<context<backend_none, device_memory>> rtn(m_size, false, alignement);
            reset();
            return rtn; 
        }

        if (!m_register && !m_on_device && m_pin && m_pin_with_device)
        { 
            resource<context<backend_none, host_memory, device_memory>> rtn(m_size, true, alignement);
            reset();
            return rtn; 
        }

        if ((!m_register && !m_on_device && !m_pin) || (!m_register && !m_on_device && m_pin && !m_pin_with_device))
        { 
            resource<context<backend_none, host_memory>> rtn(m_size, m_pin, alignement);
            reset();
            return rtn; 
        }
    }

    resource_builder reset() 
    { 
        m_on_device = false; 
        m_pin = false; 
        m_pin_with_device = false; 
        m_register = false; 
        m_alignement = 0; 
        m_size = 0;
        return *this;
    }

private:
    bool        m_on_device;
    bool        m_pin;
    bool        m_pin_with_device;
    bool        m_register;
    std::size_t m_alignement;
    std::size_t m_size;
};