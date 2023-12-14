#pragma once

/*------------------------------------------------------------------*/
/*                         Mirrored resource                        */
/*------------------------------------------------------------------*/

template <typename Resource>
class mirrored : public Resource
{
public:
    using resource_t = Resource;
    using this_type = mirrored<resource_t>;

    mirrored() = default;

    template <typename... Args>
    mirrored(Args&&... args)
      : resource_t{std::forward<Args>(args)...}
    {
    }

    mirrored(const this_type& r) noexcept = delete;

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return ptr_on_device(resource_t::allocate(size, alignment));
    }

    void* reallocate(void* ptr, std::size_t size)
    {
        if (!ptr_in_device_arena(ptr))
        {
            fmt::print("{} : [error] pointer not allocated in device arena \n", ptr);
            return nullptr;
        }
        void* tmp = ptr_on_host(ptr);
        void* tmp_reallocated = resource_t::reallocate(tmp, size);
        ptr = nullptr;
        return ptr_on_device(tmp_reallocated);
    }

    void deallocate(void* ptr, std::size_t size = 0)
    {
        if (!ptr_in_device_arena(ptr))
        {
            fmt::print("{} : [error] pointer not allocated in device arena \n", ptr);
            return;
        }
        void* tmp = ptr_on_host(ptr);
        resource_t::deallocate(tmp, size);
        ptr = nullptr;
    }

    void* ptr_on_device(void* ptr)
    {
        void* host_arena = this->get_address();
        uintptr_t ptr_ = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t host_arena_ = reinterpret_cast<uintptr_t>(host_arena);
        uintptr_t diff = ptr_ - host_arena_;
        void* rtn =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(this->get_address_device()) + diff);
        return rtn;
    }

    void* ptr_on_host(void* ptr)
    {
        uintptr_t ptr_ = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t device_arena_ = reinterpret_cast<uintptr_t>(this->get_address_device());
        uintptr_t diff = ptr_ - device_arena_;
        uintptr_t res = reinterpret_cast<uintptr_t>(this->get_address()) + diff;
        return reinterpret_cast<void*>(res);
    }

    bool ptr_in_device_arena(void* ptr)
    {
        uintptr_t ptr_ = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t device_arena_ = reinterpret_cast<uintptr_t>(this->get_address_device());
        if (ptr_ - device_arena_ < 0 || ptr_ - device_arena_ >= this->get_size())
            return false;
        return true;
    }
};

/*------------------------------------------------------------------*/
/*                      Non-mirrored resource                       */
/*------------------------------------------------------------------*/

template <typename Resource>
class simple : public Resource
{
public:
    using resource_t = Resource;
    using this_type = simple<resource_t>;

    simple()
      : resource_t{}
    {
    }

    simple(const std::size_t size, const std::size_t alignment = 0)
      : resource_t{size, alignment}
    {
    }

    simple(void* ptr, const std::size_t size)
      : resource_t{ptr, size}
    {
    }

    simple(void* ptr_a, void* ptr_b, const std::size_t size)
      : resource_t{ptr_a, ptr_b, size}
    {
    }
};
