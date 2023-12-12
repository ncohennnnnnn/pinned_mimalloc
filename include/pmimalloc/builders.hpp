#pragma once

#include <pmimalloc/base.hpp>
#include <pmimalloc/context.hpp>
#include <pmimalloc/ext_stdmalloc.hpp>
#include <pmimalloc/handler.hpp>
#include <pmimalloc/memory.hpp>
#include <pmimalloc/mirroring.hpp>
#include <pmimalloc/pinning.hpp>
#include <pmimalloc/resource.hpp>

#if PMIMALLOC_WITH_MIMALLOC
# include <pmimalloc/ext_mimalloc.hpp>
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

/*
 ------------------------------------------------------------------------
| General structure :                                                    |
|         0           1          2         3         4        5          |
|     Mirrored -- Resource -- Context -- Pinned -- Memory -- Base        |
|                     \         \                                        |
|                     Malloc     Backend                                 |
 ------------------------------------------------------------------------
*/

/*
------------------------------------------------------------------
                   Nested resource technology                     
------------------------------------------------------------------

Replace a resource class template at postion I in a nested resource type.
    Example:
        res_orig = res0<res1<res2<res3<...>, ...>, ...>, ...>
        MyReplacementResource = MyReplacementResource<NestedResource, U1, U2, ...>
        where U1, U2, ... are additional template arguments. Then :
        replace_resource_t<2, res_orig, MyReplacementResource, U1, U2, ...> -> res_new
        res_new == res0<res1<MyReplacementResource<res3<...>, U1, U2, ...>, ...>, ...>
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
------------------------------------------------------------------
                       Resource builder                           
------------------------------------------------------------------

 Default resource :
     0         1           2           3              4          5
  simple -- Resource -- Context -- not_pinned -- host_memory -- base
                     \           \
                    std_malloc  backend_none
*/

using default_resource_for_builder =
    simple<resource<context<not_pinned<host_memory<base>>, backend_none>, ext_stdmalloc>>;

template <typename Resource = default_resource_for_builder>
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
    using resource_shared_t = std::shared_ptr<resource_t>;

    constexpr resource_builder() {}

    constexpr resource_builder(const resource_builder&) = default;
    constexpr resource_builder(resource_builder&&) = default;

private:
    constexpr auto mirrored_allocations(void) const
    {
        // mirrored resources stored at position 0 in the resource nest
        return updated<0, mirrored>();
    }

public:
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

    constexpr auto pin(void) const
    {
        // pinned resources are stored at position 3 in the resource nest
        return updated<3, pinned>();
    }

#if WITH_CUDA
    constexpr auto cuda_pin(void) const
    {
        // pinned resources are stored at position 3 in the resource nest
        return updated<3, cuda_pinned>();
    }
#endif

    constexpr auto on_host(void) const
    {
        // memory resources are stored at position 4 in the resource nest
        return updated<4, host_memory>();
    }

    constexpr auto on_host_and_device(void) const
    {
        // memory resources are stored at position 4 in the resource nest
        auto tmp = mirrored_allocations();
        return tmp.template updated<4, host_device_memory>();
    }

    constexpr auto mirror_user_memory(void) const
    {
        // memory resources are stored at position 4 in the resource nest
        auto tmp = mirrored_allocations();
        return tmp.template updated<4, mirrored_user_memory>();
    }

    constexpr auto use_host_memory(void) const
    {
        // memory resources are stored at position 4 in the resource nest
        return updated<4, user_host_memory>();
    }

    constexpr resource_builder<> clear(void) const
    {
        return {};
    }

    template <typename... Args>
    constexpr resource_shared_t sbuild(Args... args) const
    {
        return std::make_shared<resource_t>(std::move(args)...);
    }

    template <typename... Args>
    constexpr resource_t build(Args... args) const
    {
        return resource_t(std::move(args)...);
    }

    template <std::size_t I, template <typename...> typename R, typename... M>
    constexpr auto updated() const
    {
        // create a new nested resource type by replacing the old resource class template
        using R_new = replace_resource_t<I, resource_t, R, M...>;
        // return new nested_resource class template instantiation
        return resource_builder<R_new>{};
    }
};

/*
------------------------------------------------------------------
                             Handler                              
------------------------------------------------------------------

 Default handler :
     0          1          2           3                4             5
  simple -- resource -- context -- not_pinned -- user_host_memory -- base
                            \
                          backend_none
*/

using default_handler = handler<context<not_pinned<user_host_memory<base>>, backend_none>>;

template <typename Handler = default_handler>
struct handler_builder
{
    using handler_t = Handler;

    constexpr handler_builder() {}

    constexpr handler_builder(const handler_builder&) = default;
    constexpr handler_builder(handler_builder&&) = default;

#if WITH_MPI || WITH_LIBFABRIC || WITH_UCX
    constexpr auto register_memory(void) const
    {
        // registered handlers are stored at position 1 in the handler nest
        return updated<1, context, backend>();
    }
#endif

    constexpr auto pin(void) const
    {
        // pinned handlers are stored at position 2 in the handler nest
        return updated<2, pinned>();
    }

#if WITH_CUDA
    constexpr auto cuda_pin(void) const
    {
        // pinned handlers are stored at position 2 in the handler nest
        return updated<2, cuda_pinned>();
    }
#endif

    constexpr handler_builder<> clear(void) const
    {
        return {};
    }

    template <typename... Args>
    constexpr handler_t build(Args... args) const
    {
        return handler_t(std::move(args)...);
    }

    template <std::size_t I, template <typename...> typename R, typename... M>
    constexpr auto updated() const
    {
        // create a new nested resource type by replacing the old resource class template
        using R_new = replace_resource_t<I, handler_t, R, M...>;
        // return new nested_resource class template instantiation
        return handler_builder<R_new>{};
    }
};

/*
------------------------------------------------------------------
             pmr::monotonic_buffer_resource builder                           
------------------------------------------------------------------

 Default resource :
     0         1           2           3              4          5
  simple -- Resource -- Context -- not_pinned -- host_memory -- base
                             \
                            backend_none
*/

// using default_pmr_resource = simple<resource<context<not_pinned<host_memory<base>>, backend_none>>>;

// template <typename Resource = default_pmr_resource>
// /** @brief Resource builder, eases the construction of a resource.
//  * @tparam Resource: the type of the resource (nested chain of resources).
//  * @tparam Args: type of parameters to feed to the resource constructor.
//  * @returns a new instance of the pmr_resource_builder class template with potentially altered template type arguments
//  * which holds an updated argument tuple.
//  * @fn build() @returns a nested resource which is constructed from `args`.
// */
// struct pmr_resource_builder
// {
//     using resource_t = Resource;

//     constexpr pmr_resource_builder() {}

//     constexpr pmr_resource_builder(const pmr_resource_builder&) = default;
//     constexpr pmr_resource_builder(pmr_resource_builder&&) = default;

// private:
//     constexpr auto mirrored_allocations(void) const
//     {
//         // mirrored resources stored at position 0 in the resource nest
//         return updated<0, mirrored>();
//     }

// public:
// #if WITH_MPI || WITH_LIBFABRIC || WITH_UCX
//     constexpr auto register_memory(void) const
//     {
//         // registered resources are stored at position 2 in the resource nest
//         return updated<2, context, backend>();
//     }
// #endif

//     constexpr auto pin(void) const
//     {
//         // pinned resources are stored at position 3 in the resource nest
//         return updated<3, pinned>();
//     }

// #if WITH_CUDA
//     constexpr auto cuda_pin(void) const
//     {
//         // pinned resources are stored at position 3 in the resource nest
//         return updated<3, cuda_pinned>();
//     }
// #endif

//     constexpr auto on_host(void) const
//     {
//         // memory resources are stored at position 4 in the resource nest
//         return updated<4, host_memory>();
//     }

//     constexpr auto on_host_and_device(void) const
//     {
//         // memory resources are stored at position 4 in the resource nest
//         auto tmp = mirrored_allocations();
//         return tmp.template updated<4, host_device_memory>();
//     }

//     constexpr auto mirror_user_memory(void) const
//     {
//         // memory resources are stored at position 4 in the resource nest
//         auto tmp = mirrored_allocations();
//         return tmp.template updated<4, mirrored_user_memory>();
//     }

//     constexpr auto use_host_memory(void) const
//     {
//         // memory resources are stored at position 4 in the resource nest
//         return updated<4, user_host_memory>();
//     }

//     constexpr pmr_resource_builder<> clear(void) const
//     {
//         return {};
//     }

//     template <typename... Args>
//     constexpr resource_t build(Args... args) const
//     {
//         return resource_t(std::move(args)...);
//     }

//     template <std::size_t I, template <typename...> typename R, typename... M>
//     constexpr auto updated() const
//     {
//         // create a new nested resource type by replacing the old resource class template
//         using R_new = replace_resource_t<I, resource_t, R, M...>;
//         // return new nested_resource class template instantiation
//         return pmr_resource_builder<R_new>{};
//     }
// };