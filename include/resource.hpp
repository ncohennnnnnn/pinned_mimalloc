#pragma once

#include <tuple>
#include <memory>

#include <resource.hpp>
#include <context.hpp>

#if WITH_MIMALLOC
#include <ex_mimalloc.hpp>
#endif
#if WITH_MPI
#include <../src/mpi/backend.hpp>
#endif
#if WITH_UCX
#include <../src/ucx/backend.hpp>
#endif
#if WITH_LIBFABRIC
#include <../src/libfabric/backend.hpp>
#endif

#include <ex_stdmalloc.hpp>
#include <backend_none.hpp>
#include <cuda_pinned.hpp>
#include <pinned.hpp>
#include <not_pinned.hpp>
#include <host_memory.hpp>
#include <device_memory.hpp>
#include <user_host_memory.hpp>
#include <user_device_memory.hpp>
#include <base.hpp>

/*------------------------------------------------------------------*/
/*                            Resource                              */
/*------------------------------------------------------------------*/

template<typename Context,typename Malloc>
class resource : public Context{
public:
    using this_type   = resource<Context, Malloc>;
    using shared_this = std::shared_ptr<resource<Context, Malloc>>;
    using malloc_t    = Malloc;
    using context_t   = Context;

    resource() noexcept 
    : m_malloc{}
    {}

    resource( context_t&& c) noexcept 
    : context_t{std::move(c)}
    , m_malloc{&c}
    {}

    resource( const this_type& r ) noexcept = delete;

    resource(std::size_t size, std::size_t alignement = 0)
    : context_t{size, alignement}
    , m_malloc{Context::m_address, Context::m_size, Context::m_numa_node}
    {}

    void* allocate(const std::size_t size, const std::size_t alignment = 0)
    {
        return m_malloc.allocate(size, alignment);
    }

    void* reallocate(void* ptr, std::size_t size) 
    {
        return m_malloc.reallocate(ptr, size); 
    }

    void deallocate(void* ptr, std::size_t size = 0)
    {
        return m_malloc.deallocate(ptr, size); 
    }

    std::size_t get_usable_size(void* ptr)
    {
        return m_malloc.get_usable_size(ptr);
    }

protected:
    malloc_t m_malloc;
};



/*------------------------------------------------------------------*/
/*                       Resource builder                           */
/*------------------------------------------------------------------*/

using default_resource = 
    resource<
        context<
            not_pinned<
                host_memory<
                    base
                >
            >
            ,
            backend_none
        >
        ,
        ex_stdmalloc
    >;

using default_args = std::tuple<
    std::tuple<>,
    std::tuple<>,
    std::tuple<>,
    std::tuple<>,
    std::tuple<>>;

// replace a resource class template at postion I in a nested resource type
// example:
// let res_orig = res0<res1<res2<res3<...>, ...>, ...>, ...>
// let MyReplacementResource = MyReplacementResource<NestedResource, U1, U2, ...>
// where U1, U2, ... are additional template arguments
// then
// replace_resource_t<2, res_orig, MyReplacementResource, U1, U2, ...> -> res_new
// res_new == res0<res1<MyReplacementResource<res3<...>, U1, U2, ...>, ...>, ...>

// primary class template declaration
template <std::size_t I, typename Nested, template<typename...> typename R, typename... M>
struct replace_resource;

// partial specialization: Nested is a class template with template paramters Inner, More...
template <std::size_t I, template <typename...> typename Nested, template<typename...> typename R, typename Inner, typename... More, typename... M>
struct replace_resource<I, Nested<Inner, More...>, R, M...> {
    // compute type recursively by decrementing I and use Inner as new Nested class template
    using type = Nested<typename replace_resource<I-1, Inner, R, M...>::type, More...>;
};

// partial specialization for I==0 (recursion end point)
template <template <typename...> typename Nested, template<typename...> typename R, typename Inner, typename... More, typename... M>
struct replace_resource<0, Nested<Inner, More...>, R, M...> {
    // inject the class template R at this point instead of Nested but keep Inner
    using type = R<Inner, M...>;
};

// helper alias to extract the member typedef `type` from the `replace_resource` struct
template <std::size_t I, typename Nested, template<typename...> typename R, typename... M>
using replace_resource_t = typename replace_resource<I, Nested, R, M...>::type;


// replace the tuple at postion I in a tuple of tuples (which stores arguments to build the nested resource)
// example:
// let tuple_orig = {{a_00, a_01, ...}, {a_10, a_11, ...}, {a_20, a_21, ...}, ...}
// let replacement_tuple = {b0, b1, ...}
// then
// replace_arg<1>(std::move(tuple_orig), std::move(replacement_tuple)) -> tuple_new
// tuple_new == {{a_00, a_01, ...}, {b0, b1, ...}, {a_20, a_21, ...}, ...}

// helper function with indices to look up elements within the tuple `args`
template<std::size_t I, typename... Ts, typename... Us, std::size_t... Is, std::size_t... Js>
constexpr inline auto replace_arg(std::tuple<Ts...>&& args, std::tuple<Us...>&& arg, std::index_sequence<Is...>, std::index_sequence<Js...>) {
    // take items 0, 1, ..., I-1, I+1, I+2, ... from `args` and replace item I with `arg`
    return std::make_tuple(std::move(std::get<Is>(args))..., std::move(arg), std::move(std::get<I+1+Js>(args))...);
}

// replace element at position I of `args` with `arg`
template<std::size_t I, typename... Ts, typename... Us>
constexpr inline auto replace_arg(std::tuple<Ts...> args, std::tuple<Us...> arg) {
    // dispatch to helper function by additionally passing indices
    return replace_arg<I>(std::move(args), std::move(arg), std::make_index_sequence<I>{}, std::make_index_sequence<sizeof...(Ts) - 1 - I>{});
}

// instantiate a neested resource from arguments in the form of a tuple of tuples 
// example:
// let nested = res0<res1<res2<...>, ...>, ...>
// let args   = {{a_00, a_01, ...}, {a_10, a_11, ...}, {a_20, a_21, ...}, ...}
// then
// nested_resource<nested>::instantiate(std::move(args)) evaluates to a concatenation of constructors:
// return 
//     res0<res1<res2<...>,...>{
//         res1<res2<...>,...>{
//             res2<...>{
//                 ...,
//                 a_20, a_21, ...},
//             a_10, a_11, ...},
//         a_00, a_01, ...};

// primary class template declaration with Index I defaulted to 0
template <typename N, std::size_t I = 0>
struct nested_resource;

// partial specialization: N is a class template with template paramters Inner, More...
template <template<typename...> typename N, typename Inner, typename... More, std::size_t I>
struct nested_resource<N<Inner, More...>, I> {

    // instantiate N<Inner, More...> with the element at postion I of the tuple
    template<typename... Ts>
    static constexpr auto instantiate(const std::tuple<Ts...>& args) {
        // dispatch to helper function by additionally passing indices enumerating the arguments within the tuple at postion I in `args`
        return instantiate(args, std::make_index_sequence<std::tuple_size_v<std::tuple_element_t<I, std::tuple<Ts...>>>>{});
    }

    // helper function with indices to look up elements within tuple extracted with std::get<I>(args)
    template<typename... Ts, std::size_t... Is>
    static constexpr auto instantiate(const std::tuple<Ts...>& args, std::index_sequence<Is...>) {
        auto arg = std::get<I>(args);
        return N<Inner, More...>{
            // first constructor argument is the next nested resource (recurse)
            nested_resource<Inner, I+1>::instantiate(std::move(args)),
            // further constructor arguments are tagen from tuple of tuples
            //std::get<Is>(std::get<I>(std::move(args)))...
            std::get<Is>(std::move(arg))...
        };
    }
};

// partial specialization when the resource is the base (recursion end point)
template<std::size_t I>
struct nested_resource<base, I> {

    // return a default constructed base
    template<typename... Ts>
    static constexpr auto instantiate(const std::tuple<Ts...>& args) {
        return base{};
    }
};

// resource_builder class template
// template type arguments:
//   - Resource: the type of the resource (nested chain of resources)
//   - Args: type of arguments to construct the nested resource (tuple of tuples)
// member functions (apart from build())
//   - return a new instance of the resource_builder class template with potentially altered template type arguments
//   - which holds an updated argument tuple
// the build() member function
//   - returns a nested resource
//   - which is constructed from the `args` tuple of tuples
template<typename Resource = default_resource, typename Args = default_args>
struct resource_builder {

    using resource_t = Resource;
    using args_t = Args;

    // constexpr resource_builder() noexcept = default;
    constexpr resource_builder() : args{} {} 
    constexpr resource_builder(args_t&& a) : args{std::move(a)} {}
    constexpr resource_builder(const resource_builder&) = default;
    constexpr resource_builder(resource_builder&&) = default;

//    0           1         2         3        4
// Resource -- Context -- Pinned -- Memory -- Base
//          \         \ 
//          Malloc     Backend
//__________________________________________________________
// Default structure :
//    0           1           2             3           4
// Resource -- Context -- not_pinned -- host_memory -- base
//          \         \ 
//      ex_stdmalloc  backend_none

#if WITH_MIMALLOC
    constexpr auto use_mimalloc(void) const {
        // arena resources are stored at position 0 in the resource nest
        return updated<0, resource, ex_mimalloc>(std::tuple<>{});
    }
#endif

    constexpr auto use_stdmalloc(void) const {
        // arena resources are stored at position 0 in the resource nest
        return updated<0, resource, ex_stdmalloc>(std::tuple<>{});
    }

    // template<typename Backend>
    constexpr auto register_memory(void) const { 
        // registered resources are stored at position 1 in the resource nest
        return updated<1, context, backend>(std::tuple<>{});
    }

    // template<typename Backend>
    constexpr auto no_register_memory(void) const {
        // registered resources are stored at position 1 in the resource nest
        return updated<1, context, backend>(std::tuple<>{});
    }

    constexpr auto pin(void) const {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, pinned>(std::tuple<>{});
    }

#if WITH_CUDA
    constexpr auto pin_cuda(void) const {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, cuda_pinned>(std::tuple<>{});
    }
#endif

    constexpr auto no_pin(void) const {
        // pinned resources are stored at position 2 in the resource nest
        return updated<2, not_pinned>(std::tuple<>{});
    }

    constexpr auto on_device(const std::size_t size, const int alignement = 0) const {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, device_memory>(std::make_tuple(size, alignement));
    }

    constexpr auto on_host(const std::size_t size, const int alignement = 0) const {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, host_memory>(std::make_tuple(size, alignement));
    }

    constexpr auto use_host_memory(void* ptr, const std::size_t size) const {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, user_host_memory>(std::make_tuple(ptr, size));
    }

    constexpr auto use_device_memory(void* ptr, const std::size_t size) const {
        // memory resources are stored at position 3 in the resource nest
        return updated<3, user_device_memory>(std::make_tuple(ptr, size));
    }

    constexpr resource_t build() const { return nested_resource<resource_t>::instantiate(args); }

    // constexpr auto build_any() const { return any_resource{build()}; }

private:
    const args_t args;

    template<std::size_t I, template<typename...> typename R, typename... M, typename Arg>
    constexpr auto updated(Arg arg) const {
        // create a new nested resource type by replacing the old resource class template
        using R_new = replace_resource_t<I, resource_t, R, M...>;
        // create new arguments by replacing the old argument tuple
        auto args_new = replace_arg<I>(args, arg);
        // return new resource_builder class template instantiation
        return resource_builder<R_new, decltype(args_new)>{std::move(args_new)};
    }
};

// inline constexpr auto resource_builder() { return resource_builder<>{}; }

inline auto host_resource(std::size_t size) {
    static constexpr auto b = resource_builder().pin().use_mimalloc();
    return b.on_host(size).build();
}

inline auto host_resource(void* ptr, std::size_t size) {
    static constexpr auto b = resource_builder().pin();
    return b.use_host_memory(ptr, size).build();
};