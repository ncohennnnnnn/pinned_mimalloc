#pragma once

#include <memory>
#include <tuple>

#include <context.hpp>
#include <resource.hpp>

#if WITH_MIMALLOC
# include <ex_mimalloc.hpp>
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
#include <device_memory.hpp>
#include <ex_stdmalloc.hpp>
#include <host_memory.hpp>
#include <not_pinned.hpp>
#include <pinned.hpp>
#include <user_device_memory.hpp>
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

    resource(const std::size_t size, const std::size_t alignement = 0)
      : context_t{size, alignement}
      , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    {
    }
    // resource(void* ptr, const std::size_t size)
    // : context_t{ptr, size}
    // , m_malloc{context_t::m_address, context_t::m_size, context_t::m_numa_node}
    // {}

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

    // std::size_t get_usable_size()
    // {
    //     return m_malloc.get_usable_size();
    // }

protected:
    malloc_t m_malloc;
};

/*------------------------------------------------------------------*/
/*                       Resource builder                           */
/*------------------------------------------------------------------*/

using default_resource =
    resource<context<not_pinned<host_memory<base>>, backend_none>, ex_stdmalloc>;

using default_args = std::tuple<>;

/*--------------------------------------------------------------------------------
  Replace a resource class template at postion I in a nested resource type.
  Example:
    res_orig = res0<res1<res2<res3<...>, ...>, ...>, ...>
    MyReplacementResource = MyReplacementResource<NestedResource, U1, U2, ...>
    where U1, U2, ... are additional template arguments. Then :
    replace_resource_t<2, res_orig, MyReplacementResource, U1, U2, ...> -> res_new
    res_new == res0<res1<MyReplacementResource<res3<...>, U1, U2, ...>, ...>, ...>
  --------------------------------------------------------------------------------*/

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

template <typename Resource = default_resource, typename Args = default_args>
/** @brief Resource builder, eases the construction of a resource.
 * @tparam Resource: the type of the resource (nested chain of resources).
 * @tparam Args: type of parameters to feed to the resource constructor.
 * @returns a new instance of the resource_builder class template with potentially altered template type arguments
 * which holds an updated argument tuple.
 * @fn build() @returns a nested resource which is constructed from the `args` tuple.
*/
struct resource_builder
{
    using resource_t = Resource;
    using resource_shared_t = std::shared_ptr<Resource>;
    using args_t = Args;

    // constexpr resource_builder() noexcept = default;
    constexpr resource_builder()
      : args{}
    {
    }
    constexpr resource_builder(args_t&& a)
      : args{std::move(a)}
    {
    }
    constexpr resource_builder(const resource_builder&) = default;
    constexpr resource_builder(resource_builder&&) = default;

    //    0           1         2         3        4
    // Resource -- Context -- Pinned -- Memory -- Base
    //          \         \                                    .
    //          Malloc     Backend
    //__________________________________________________________
    // Default structure :
    //    0           1           2             3           4
    // Resource -- Context -- not_pinned -- host_memory -- base
    //          \         \                                    .
    //      ex_stdmalloc  backend_none

#if WITH_MIMALLOC
    constexpr auto use_mimalloc(void) const
    {
        // arena resources are stored at position 0 in the resource nest
        return updated<0, resource, ex_mimalloc>(std::tuple<>{});
    }
#endif

    constexpr auto use_stdmalloc(void) const
    {
        // arena resources are stored at position 0 in the resource nest
        return updated<0, resource, ex_stdmalloc>(std::tuple<>{});
    }

#if WITH_MPI || WITH_LIBFABRIC || WITH_UCX
    constexpr auto register_memory(void) const
    {
        // registered resources are stored at position 1 in the resource nest
        return updated<1, context, backend>(std::tuple<>{});
    }
#endif

    constexpr auto no_register_memory(void) const
    {
        // registered resources are stored at position 1 in the resource nest
        return updated<1, context, backend_none>(std::tuple<>{});
    }

    constexpr auto pin(void) const
    {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, pinned>(std::tuple<>{});
    }

#if WITH_CUDA
    constexpr auto pin_cuda(void) const
    {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, cuda_pinned>(std::tuple<>{});
    }
#endif

    constexpr auto no_pin(void) const
    {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, not_pinned>(std::tuple<>{});
    }

    constexpr auto on_device(const std::size_t size, const int alignement = 0) const
    {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, device_memory>(std::make_tuple(size, alignement));
    }

    constexpr auto on_host(const std::size_t size, const int alignement = 0) const
    {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, host_memory>(std::make_tuple(size, alignement));
    }

    constexpr auto use_host_memory(void* ptr, const std::size_t size) const
    {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, user_host_memory>(std::make_tuple(ptr, size));
    }

    constexpr auto use_device_memory(void* ptr, const std::size_t size) const
    {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, user_device_memory>(std::make_tuple(ptr, size));
    }

    constexpr resource_shared_t sbuild() const
    {
        return sbuild(std::make_index_sequence<std::tuple_size_v<args_t>>{});
    }

    template <std::size_t... I>
    constexpr resource_shared_t sbuild(std::index_sequence<I...>) const
    {
        return std::make_shared<resource_t>(std::get<I>(args)...);
    }

    constexpr resource_t build() const
    {
        return build(std::make_index_sequence<std::tuple_size_v<args_t>>{});
    }

    template <std::size_t... I>
    constexpr resource_t build(std::index_sequence<I...>) const
    {
        return resource_t(std::get<I>(args)...);
    }

    // constexpr auto build_any() const { return any_resource{build()}; }

private:
    const args_t args;

    template <std::size_t I, template <typename...> typename R, typename... M, typename Arg>
    constexpr auto updated(Arg arg) const
    {
        // create a new nested resource type by replacing the old resource class template
        using R_new = replace_resource_t<I, resource_t, R, M...>;
        // return new resource_builder class template instantiation
        return resource_builder<R_new, Arg>{std::move(arg)};
    }
};

// template<typename Resource_Builder>
// struct Res{
//     using type  = decltype(Resource_Builder::build());
//     using stype = std::shared_ptr<type>;

//     Res(Resource_Builder&& rb)
//     : obj{rb.build()}
//     , sptr{std::make_shared<type>(obj)}
//     { }

//     type  obj;
//     stype sptr;
// };
