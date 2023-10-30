#pragma once 

#include <resource.hpp>

class resource_builder {
public:
    void m_on_device() { m_on_device = true; }

    void pin() { m_pin = true; }

    void pin_with_device() { m_pin_with_device = true; }

    void register() { m_register = true; }

    void set_size(std::size_t size) { m_size = size; }

    void alignement(std::size_t size) { m_alignement = size; }

    resource build() {
        if (m_register && m_on_device)
        { return resource<context<backend, device_memory>>{m_size, false, alignement}; }

        if (m_register && !m_on_device && m_pin && m_pin_with_device)
        { return resource<context<backend, host_memory, device_memory>>{m_size, true, alignement}; }

        if ((m_register && !m_on_device && m_pin && !m_pin_with_device) || (m_register && !m_on_device && !m_pin))
        { return resource<context<backend, host_memory>>{m_size, m_pin, alignement}; }

        if (!m_register && m_on_device)
        { return resource<context<backend_none, device_memory>>{m_size, false, alignement}; }

        if (!m_register && !m_on_device && m_pin && m_pin_with_device)
        { return resource<context<backend_none, host_memory, device_memory>>{m_size, true, alignement}; }

        if ((!m_register && !m_on_device && !m_pin) || (!m_register && !m_on_device && m_pin && !m_pin_with_device))
        { return resource<context<backend_none, host_memory>>{m_size, m_pin, alignement}; }
    }

private:
    bool m_on_device = false;
    bool m_pin = false;
    bool m_pin_with_device = false;
    bool m_register = false;
    std::size_t m_alignement = 0;
    std::size_t m_size = 0;
};