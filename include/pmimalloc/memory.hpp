#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
//
#include <cuda_runtime.h>
#include <fmt/core.h>
//
#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
# define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif
// #include <pmimalloc/numa.hpp>

// /* TODO: Steal numa stuff from Fabian */
// int get_node(void* ptr){
//     int numa_node[1] = {-1};
//     void* page = (void*)((std::size_t)ptr & ~((std::size_t)getpagesize()-1));
//     int err = move_pages(getpid(), 1, &page , NULL, numa_node, 0);
//     if (err == -1) {
//         fmt::print("Move page failed from get_node(). \n");
//         return -1;
//     }
//     return numa_node[0];
// }

/*------------------------------------------------------------------*/
/*                      To-be-allocated memory                      */
/*------------------------------------------------------------------*/

template <typename Base>
class to_be_allocated_memory : public Base
{
protected:
    void _host_alloc(const std::size_t alignment = 0)
    {
        _align_and_set_total_size(alignment);

        m_raw_address =
            mmap(0, m_total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m_raw_address == MAP_FAILED)
        {
            std::cerr << "[error] mmap failed (error thrown) \n" << std::endl;
            m_raw_address = nullptr;
            return;
        }
        else
        {
            /* Tell the OS that this memory can be considered for huge pages */
            madvise(m_raw_address, m_total_size, MADV_HUGEPAGE);
        }
        fmt::print("{} : Host memory of size {} mmaped \n", m_raw_address, m_total_size);

        if (m_raw_address == nullptr)
        {
            fmt::print("[error] Host allocation failed \n");
            return;
        }

        this->set_address(_align(m_raw_address, alignment));
        fmt::print("{} : Aligned host pointer \n", this->m_address);
    }

    void _device_alloc(const std::size_t /* alignment */ = 0)
    {
        cudaMalloc(&m_address_device, this->m_size);
        fmt::print(
            "{} : Device memory of size {} cudaMallocated \n", m_address_device, this->m_size);

        if (m_address_device == nullptr)
        {
            fmt::print("[error] Device allocation failed \n");
            return;
        }
    }

    void _mirror_alloc(const std::size_t alignment, size_t /*size*/)
    {
        _align_and_set_total_size(alignment);
        _host_alloc(alignment);
        _device_alloc(alignment);
    }

    void _host_dealloc(void)
    {
        if (munmap(m_raw_address, m_total_size) != 0)
        {
            fmt::print("{} : [error] munmap failed \n", m_raw_address);
            return;
        }
        fmt::print("{} : Host memory munmap'ed \n", m_raw_address);
    }

    void _device_dealloc(void)
    {
        cudaFree(m_address_device);
        fmt::print("{} : Device memory cudaFreed \n", m_address_device);
    }

    void _align_and_set_total_size(const std::size_t alignment)
    {
        if (alignment == 0)
        {
            m_total_size = this->m_size;
            return;
        }
        else if ((alignment & (alignment - 1)) != 0)
        {
            fmt::print("[error] null or odd alignement! \n");
            return;
        }

        set_total_size(this->m_size + alignment - 1);
    }

    /* Calculate the aligned pointer within the allocated memory block. */
    void* _align(void* ptr, const std::size_t alignment)
    {
        if (alignment == 0)
            return ptr;
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;
        void* rtn = reinterpret_cast<void*>(aligned_ptr);
        return rtn;
    }

    void set_raw_address(void* ptr)
    {
        m_raw_address = ptr;
    }

    void set_address_device(void* ptr)
    {
        m_address_device = ptr;
    }

    void set_total_size(std::size_t s)
    {
        m_total_size = s;
    }

    void* m_raw_address = nullptr;
    void* m_address_device = nullptr;
    std::size_t m_total_size = 0;
};

/*------------------------------------------------------------------*/
/*                           Host memory                            */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory on the host.
*/
class host_memory : private to_be_allocated_memory<Base>
{
public:
    host_memory() = default;

    host_memory(const std::size_t size, const std::size_t /* alignment */ = 0)
    {
        this->set_size(size);
        this->_host_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE);
    }

    ~host_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
            this->_host_dealloc();
        else
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", this->m_raw_address);
    }

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }
};

/*------------------------------------------------------------------*/
/*                       Host-device memory                         */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory mirrored on the host and device.
 * TODO:  do we need to know the device id ?
*/
class host_device_memory : private to_be_allocated_memory<Base>
{
public:
    host_device_memory() = default;

    host_device_memory(const std::size_t size, const std::size_t /* alignment */ = 0)
    {
        this->set_size(size);
        this->_mirror_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
    }

    ~host_device_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
            this->_host_dealloc();
        else
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", this->m_raw_address);

        this->_device_dealloc();
    }

    void* get_address_device(void)
    {
        return this->m_address_device;
    }

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }
};

/*------------------------------------------------------------------*/
/*                         User host memory                         */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Already allocated memory living on the host. */
class user_host_memory : public Base
{
public:
    user_host_memory() = default;

    user_host_memory(void* ptr, const std::size_t s)
    {
        this->set_address(ptr);
        this->set_size(s);
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
    }

    ~user_host_memory() {}

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size();
    }

    int get_numa_node(void)
    {
        return this->m_numa_node();
    }
};

/*------------------------------------------------------------------*/
/*                       Mirrored user memory                       */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory mirrored on the host and device.
 * TODO:  do we need to know the device id ?
*/
class mirrored_user_memory : private to_be_allocated_memory<Base>
{
public:
    mirrored_user_memory() = default;

    mirrored_user_memory(void* ptr, const std::size_t s)
    {
        this->set_size(s);
        if (!_is_on_device(ptr))
        {
            this->_device_alloc(s);
            if (!_is_aligned(ptr))
            {
                fmt::print("[error] Pointer not aligned, cannot be used for a mimalloc arena !\n");
                mirrored_user_memory{};
            }
            this->set_address(ptr);
            this->set_raw_address(ptr);
            m_from_host = true;
        }
        else
        {
            this->_host_alloc(s);
            this->set_address_device(ptr);
            m_from_device = true;
        }
    }

    mirrored_user_memory(void* ptr_a, void* ptr_b, const std::size_t s)
      : m_from_device{true}
      , m_from_host{true}
    {
        this->set_size(s);
        if (_is_on_device(ptr_a) == _is_on_device(ptr_b))
        {
            fmt::print("[error] Both pointers live on the same kind of memory !\n");
            mirrored_user_memory{};
        }
        if (_is_on_device(ptr_b))
        {
            this->set_address(ptr_a);
            this->set_address_device(ptr_b);
            this->set_raw_address(ptr_a);
        }
    }

    ~mirrored_user_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
        {
            if (!m_from_host)
                this->_host_dealloc();
        }
        else
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", this->m_raw_address);
        if (!m_from_device)
            this->_device_dealloc();
    }

    void* get_address_device(void)
    {
        return this->m_address_device;
    }

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }

private:
    bool _is_on_device(const void* ptr)
    {
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err || attributes.type != 2)
            return false;
        return true;
    }

    bool _is_aligned(void* ptr)
    {
        return (ptr == this->_align(ptr, MIMALLOC_SEGMENT_ALIGNED_SIZE));
    }

    bool m_from_host = false;
    bool m_from_device = false;
};

/*------------------------------------------------------------------*/
/*                        pmr::Host memory                          */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory on the host.
*/
class pmr_host_memory : private to_be_allocated_memory<Base>
{
public:
    pmr_host_memory() = default;

    pmr_host_memory(const std::size_t size, const std::size_t alignment = 0)
    {
        this->set_size(size);
        this->_host_alloc(0);
    }

    ~pmr_host_memory() = default;

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }
};

/*------------------------------------------------------------------*/
/*                     pmr::Host-device memory                      */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory mirrored on the host and device.
 * TODO:  do we need to know the device id ?
*/
class pmr_host_device_memory : private to_be_allocated_memory<Base>
{
public:
    pmr_host_device_memory() = default;

    pmr_host_device_memory(const std::size_t size, const std::size_t alignment = 0)
    {
        this->set_size(size);
        this->_mirror_alloc(0, size);
    }

    ~pmr_host_device_memory()
    {
        this->_device_dealloc();
    }

    void* get_address_device(void)
    {
        return this->m_address_device;
    }

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }
};

/*------------------------------------------------------------------*/
/*                       Mirrored user memory                       */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory mirrored on the host and device.
 * TODO:  do we need to know the device id ?
*/
class pmr_mirrored_user_memory : private to_be_allocated_memory<Base>
{
public:
    pmr_mirrored_user_memory() = default;

    pmr_mirrored_user_memory(void* ptr, const std::size_t s)
    {
        this->set_size(s);
        if (!_is_on_device(ptr))
        {
            this->_device_alloc(s);
            this->set_address(ptr);
            this->set_raw_address(ptr);
            m_from_host = true;
        }
        else
        {
            this->_host_alloc(s);
            this->set_address_device(ptr);
            m_from_device = true;
        }
    }

    pmr_mirrored_user_memory(void* ptr_a, void* ptr_b, const std::size_t s)
      : m_from_device{true}
      , m_from_host{true}
    {
        this->set_size(s);
        if (_is_on_device(ptr_a) == _is_on_device(ptr_b))
        {
            fmt::print("[error] Both pointers live on the same kind of memory !\n");
            pmr_mirrored_user_memory{};
        }
        if (_is_on_device(ptr_b))
        {
            this->set_address(ptr_a);
            this->set_address_device(ptr_b);
            this->set_raw_address(ptr_a);
        }
    }

    ~pmr_mirrored_user_memory()
    {
        if (!m_from_device)
            this->_device_dealloc();
    }

    void* get_address_device(void)
    {
        return this->m_address_device;
    }

    void* get_address(void)
    {
        return this->m_address;
    }

    std::size_t get_size(void)
    {
        return this->m_size;
    }

    int get_numa_node(void)
    {
        return this->m_numa_node;
    }

private:
    bool _is_on_device(const void* ptr)
    {
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err || attributes.type != 2)
            return false;
        return true;
    }

    bool m_from_host = false;
    bool m_from_device = false;
};
