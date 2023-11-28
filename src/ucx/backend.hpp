#include <fmt>

#include <hwmalloc/heap.hpp>
#include <hwmalloc/register.hpp>
#include <hwmalloc/register_device.hpp>

#include <oomph/config.hpp>

// paths relative to backend
#include <region.hpp>

class rma_context
{
public:
    using region_type = rma_region;
    using device_region_type = rma_region;

private:
    ucp_context_h m_context;

public:
    rma_context()
      : m_heap{this}
    {
    }
    rma_context(context_impl const&) = delete;
    rma_context(context_impl&&) = delete;

    rma_region make_region(void* ptr, std::size_t size, bool gpu = false)
    {
        return {m_context, ptr, size, gpu};
    }

    template <>
    inline rma_region register_memory<rma_context>(rma_context& c, void* ptr, std::size_t size)
    {
        return c.make_region(ptr, size);
    }

#if OOMPH_ENABLE_DEVICE
    template <>
    inline rma_region
    register_device_memory<rma_context>(rma_context& c, void* ptr, std::size_t size)
    {
        return c.make_region(ptr, size, true);
    }
#endif
