#pragma once

#include <memory>
#include <memory_resource>
#include <tuple>

/*------------------------------------------------------------------*/
/*                     monotonic_buffer_resource                    */
/*------------------------------------------------------------------*/

template <typename Base>
/** @brief Memory living on the host. */
class monotonic_buffer : public std::pmr::
{
public:
    host_memory() = default;

    host_memory(const std::size_t size, const std::size_t alignment = 0)
    {
        _allocate(size, alignment);
    }

    ~host_memory()
    {
#ifndef MI_SKIP_COLLECT_ON_EXIT
        int val = 0;    // mi_option_get(mi_option_limit_os_alloc);
#endif
        if (!val)
        {
            _deallocate();
        }
        else
        {
            fmt::print("{} : Skipped std::free (mi_option_limit_os_alloc) \n", m_address);
        }
    }

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

private:
    void _allocate(const std::size_t size, const std::size_t alignment = 0)
    {
#if PMIMALLOC_WITH_MIMALLOC
        _aligned_alloc(MIMALLOC_SEGMENT_ALIGNED_SIZE, size);
#else
        if (alignment != 0)
        {
            _aligned_alloc(alignment, size);
        }
        else
        {
            m_address = std::malloc(size);
            m_raw_address = m_address;
            m_total_size = size;
            fmt::print("{} : Host memory of size {} std::mallocated \n", m_address, size);
        }
#endif
        // numa_tools n;
        // m_numa_node = numa_tools::get_node(m_address);
        // fmt::print("{} : Pointer is on numa node : {} \n", m_address, m_numa_node);
        m_numa_node = -1;
        m_size = size;
    }

    void _deallocate()
    {
#ifdef PMIMALLOC_USE_MMAP
        if (munmap(m_raw_address, m_total_size) != 0)
        {
            std::cerr << "munmap failed \n" << std::endl;
        }
        fmt::print("{} : Host memory munmaped \n", m_raw_address);
#else
        std::free(m_raw_address);
        fmt::print("{} : Host memory std::freed \n", m_raw_address);
#endif
    }

    void _aligned_alloc(const std::size_t alignment, const std::size_t size)
    {
        /* Check the alignment, must be a power of 2 and non-zero. */
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            return;

        size_t total_size = size + alignment - 1;
        m_total_size = total_size;

        /* Allocate on host */
        _aligned_host_alloc(alignment);
    }

    void _aligned_host_alloc(const std::size_t alignment)
    {
#ifdef PMIMALLOC_USE_MMAP
        void* original_ptr =
            mmap(0, m_total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (original_ptr == MAP_FAILED)
        {
            fmt::print("[error] mmap failed (error thrown) \n");
            original_ptr = nullptr;
        }
        else
        {
            // tell the OS that this memory can be considered for huge pages
            madvise(original_ptr, m_total_size, MADV_HUGEPAGE);
        }
        fmt::print("{} : Host memory of size {} mmaped \n", original_ptr, m_total_size);
#else
        void* original_ptr = std::malloc(m_total_size);
        fmt::print("{} : Host memory of size {} std::mallocated \n", original_ptr, m_total_size);
#endif

        if (original_ptr == nullptr)
        {
            fmt::print("[error] mmap failed (nullptr) \n");
            return;
        }

        m_raw_address = original_ptr;

        // Calculate the aligned pointer within the allocated memory block.
        uintptr_t unaligned_ptr = reinterpret_cast<uintptr_t>(m_raw_address);
        uintptr_t misalignment = unaligned_ptr % alignment;
        uintptr_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);
        uintptr_t aligned_ptr = unaligned_ptr + adjustment;
        // Store the original pointer before the aligned pointer.
        //        uintptr_t* ptr_storage  = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        //        *ptr_storage = reinterpret_cast<uintptr_t>(original_ptr);

        m_address = reinterpret_cast<void*>(aligned_ptr);
        fmt::print("{} : Aligned host pointer \n", m_address);
    }

protected:
    void* m_address;
    void* m_raw_address;
    std::size_t m_size;
    std::size_t m_total_size;
    int m_numa_node;
};
