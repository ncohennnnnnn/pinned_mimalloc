#pragma once

#include <memory>
#include <tuple>

#include <context.hpp>
#include <resource.hpp>

#if PMIMALLOC_WITH_MIMALLOC
# include <ext_mimalloc.hpp>
#endif
#if WITH_MPI
# include <../src/mpi/backend.hpp>
#endif
#if WITH_UCX
# include <../src/ucx/backend.hpp>
#endif
#if WITH_LIBFABRIC
# include <../src/libfabric/backend.hpp>
#endif

#include <backend_none.hpp>
#include <base.hpp>
#include <cuda_pinned.hpp>
#include <ext_stdmalloc.hpp>
#include <host_memory.hpp>
#include <mirror_memory.hpp>
#include <mirrored.hpp>
#include <not_pinned.hpp>
#include <pinned.hpp>
#include <simple.hpp>
#include <user_host_memory.hpp>

/*------------------------------------------------------------------*/
/*                            Resource                              */
/*------------------------------------------------------------------*/

template <typename Context, typename Malloc>
class resource : public Context
{
public:
    using this_type = resource<Context, Malloc>;
    using shared_this = std::shared_ptr<resource<Context, Malloc>>;
    using malloc_t = Malloc;
    using context_t = Context;

    resource() noexcept
      : m_malloc{}
    {
    }

    resource(context_t&& c) noexcept
      : context_t{std::move(c)}
      , m_malloc{&c}
    {
    }

    resource(const this_type& r) noexcept = delete;

    resource(const std::size_t size, const std::size_t alignment = 0)
      : context_t{size, alignment}
      , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    {
    }

    resource(void* ptr, const std::size_t size)
      : context_t{ptr, size}
      , m_malloc{ptr, size, context_t::m_numa_node}
    {
    }

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return m_malloc.allocate(size, alignment);
    }

    void* reallocate(void* ptr, const std::size_t size)
    {
        return m_malloc.reallocate(ptr, size);
    }

    void deallocate(void* ptr, const std::size_t size = 0)
    {
        return m_malloc.deallocate(ptr, size);
    }

protected:
    malloc_t m_malloc;
};

/*------------------------------------------------------------------*/
/*                       Resource builder                           */
/*------------------------------------------------------------------*/

using default_resource =
    simple<resource<context<not_pinned<host_memory<base>>, backend_none>, ext_stdmalloc>>;

using default_args = std::tuple<>;

/*
--------------------------------------------------------------------------------------
Replace a resource class template at postion I in a nested resource type.
    Example:
        res_orig = res0<res1<res2<res3<...>, ...>, ...>, ...>
        MyReplacementResource = MyReplacementResource<NestedResource, U1, U2, ...>
        where U1, U2, ... are additional template arguments. Then :
        replace_resource_t<2, res_orig, MyReplacementResource, U1, U2, ...> -> res_new
        res_new == res0<res1<MyReplacementResource<res3<...>, U1, U2, ...>, ...>, ...>
--------------------------------------------------------------------------------------
*/

// primary class template declaration
template <std::size_t I, typename Nested, template <typename...> typename R, typename... M>
struct replace_resource;

// partial specialization: Nested is a class template with template paramters Inner, More...
template <std::size_t I, template <typename...> typename Nested, template <typename...> typename R,
    typename Inner, typename... More, typename... M>
struct replace_resource<I, Nested<Inner, More...>, R, M...>
{
    // compute type recursively by decrementing I and use Inner as new Nested class template
    using type = Nested<typename replace_resource<I - 1, Inner, R, M...>::type, More...>;
};

// partial specialization for I==0 (recursion end point)
template <template <typename...> typename Nested, template <typename...> typename R, typename Inner,
    typename... More, typename... M>
struct replace_resource<0, Nested<Inner, More...>, R, M...>
{
    // inject the class template R at this point instead of Nested but keep Inner
    using type = R<Inner, M...>;
};

// helper alias to extract the member typedef `type` from the `replace_resource` struct
template <std::size_t I, typename Nested, template <typename...> typename R, typename... M>
using replace_resource_t = typename replace_resource<I, Nested, R, M...>::type;

/*
 ------------------------------------------------------------------------
| General structure :                                                    |
|         0           1          2         3         4        5          |
|     Mirrored -- Resource -- Context -- Pinned -- Memory -- Base        |
|                     \         \                                        |
|                     Malloc     Backend                                 |
|                                                                        |
| Default structure :                                                    |
|        0         1           2           3             4           5   |
|     simple -- resource -- context -- not_pinned -- host_memory -- base |
|                     \         \                                        |
|              ext_std_malloc  backend_none                              |
 ------------------------------------------------------------------------
*/

template <typename Resource = default_resource>
/** @brief Resource builder, eases the construction of a resource.
 * @tparam Resource: the type of the resource (nested chain of resources).
 * @tparam Args: type of parameters to feed to the resource constructor.
 * @returns a new instance of the resource_builder class template with potentially altered template type arguments
 * which holds an updated argument tuple.
 * @fn build() @returns a nested resource which is constructed from `args`.
*/
struct resource_builder
{
    using resource_t = Resource;
    using resource_shared_t = std::shared_ptr<Resource>;

    constexpr resource_builder() {}

    constexpr resource_builder(const resource_builder&) = default;
    constexpr resource_builder(resource_builder&&) = default;

    constexpr auto is_simple(void) const
    {
        // mirrored resources stored at position 0 in the resource nest
        return updated<0, simple>();
    }

    constexpr auto is_mirrored(void) const
    {
        // mirrored resources stored at position 0 in the resource nest
        return updated<0, mirrored>();
    }

#if PMIMALLOC_WITH_MIMALLOC
    constexpr auto use_mimalloc(void) const
    {
        // arena resources are stored at position 1 in the resource nest
        return updated<1, resource, ext_mimalloc>();
    }
#endif

    constexpr auto use_stdmalloc(void) const
    {
        // arena resources are stored at position 1 in the resource nest
        return updated<1, resource, ext_stdmalloc>();
    }

#if WITH_MPI || WITH_LIBFABRIC || WITH_UCX
    constexpr auto register_memory(void) const
    {
        // registered resources are stored at position 2 in the resource nest
        return updated<2, context, backend>();
    }
#endif

    constexpr auto no_register_memory(void) const
    {
        // registered resources are stored at position 2 in the resource nest
        return updated<2, context, backend_none>();
    }

    constexpr auto pin(void) const
    {
        // pinned resources are stored at position 3 in the resource nest
        return updated<3, pinned>();
    }

#if WITH_CUDA
    constexpr auto pin_cuda(void) const
    {
        // pinned resources are stored at position 3 in the resource nest
        return updated<3, cuda_pinned>();
    }
#endif

    constexpr auto no_pin(void) const
    {
        // pinned resources are stored at position 3 in the resource nest
        return updated<3, not_pinned>();
    }

    // constexpr auto on_device(const std::size_t size, const int alignment = 0) const
    // {
    //     // memory resources are stored at position 4 in the resource nest
    //     return updated<4, device_memory>(std::make_tuple(size, alignment));
    // }

    constexpr auto on_host() const
    {
        // memory resources are stored at position 4 in the resource nest
        return updated<4, host_memory>();
    }

    constexpr auto on_mirror() const
    {
        // memory resources are stored at position 4 in the resource nest
        auto tmp = is_mirrored();
        return tmp.template updated<4, mirror_memory>();
    }

    constexpr auto use_host_memory(/*void* ptr, const std::size_t size*/) const
    {
        // memory resources are stored at position 4 in the resource nest
        return updated<4, user_host_memory>();
    }

    // constexpr auto use_device_memory(void* ptr, const std::size_t size) const
    // {
    //     // memory resources are stored at position 4 in the resource nest
    //     return updated<4, user_device_memory>(std::make_tuple(ptr, size));
    // }

    template <typename... Args>
    constexpr resource_shared_t sbuild(Args... a) const
    {
        return std::make_shared<resource_t>(std::move(a)...);
    }

    template <typename... Args>
    constexpr resource_t build(Args... a) const
    {
        return resource_t(std::move(a)...);
    }

    template <std::size_t I, template <typename...> typename R, typename... M, typename... Arg>
    constexpr auto updated(Arg... arg) const
    {
        // create a new nested resource type by replacing the old resource class template
        using R_new = replace_resource_t<I, resource_t, R, M...>;
        // return new resource_builder class template instantiation
        return resource_builder<R_new, Arg...>{std::move(arg)...};
    }
};