#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

// Shared thread-local pointer
// A indexed_tl_ptr
// - shares its _instance_idx with all of its copies
// - retrains ownership of a thread_local object of type T, which is shared among all of its copies
//   that live on the same thread
// - objects of type T are created on demand through a maker class
// - managed thread_local objects of type T are destroyed when the thread is destroyed
template <typename T>
class indexed_tl_ptr
{
public:
    using element_type = std::remove_extent_t<T>;
    using delete_fn_type = std::function<void(element_type*)>;

private:
    std::size_t _instance_idx;
    std::function<element_type*()> _maker;
    delete_fn_type _deleter;

private:
    std::atomic<std::size_t>& idx()
    {
        static std::atomic<std::size_t> _i{1};
        return _i;
    }

public:
    indexed_tl_ptr() noexcept
      : _instance_idx{0}
    {
    }

    template <typename Maker, typename Deleter>
    indexed_tl_ptr(Maker m, Deleter d)
      : _instance_idx{idx().fetch_add(1)}
      , _maker{std::move(m)}
      , _deleter{std::move(d)}
    {
    }

    indexed_tl_ptr(indexed_tl_ptr const& other) = default;
    indexed_tl_ptr(indexed_tl_ptr&& other) = default;
    indexed_tl_ptr& operator=(indexed_tl_ptr const& other) = default;
    indexed_tl_ptr& operator=(indexed_tl_ptr&& other) = default;

    using ptype = std::unique_ptr<element_type, delete_fn_type>;
    element_type* get()
    {
        static thread_local std::vector<ptype> _tl;
        if (!_instance_idx)
            return nullptr;
        if (_tl.size() < _instance_idx)
            _tl.resize(_instance_idx, nullptr);
        auto& i = _tl[_instance_idx - 1u];
        if (!i)
            i = ptype{_maker(), _deleter};
        return i.get();
    }

    T& operator*()
    {
        return static_cast<T&>(*get());
    }

    T* operator->()
    {
        return static_cast<T*>(get());
    }

    operator bool() const noexcept
    {
        return (bool) _instance_idx;
    }

    void swap(indexed_tl_ptr& other) noexcept
    {
        using namespace std;
        swap(_instance_idx, other._instance_idx);
        swap(_maker, other._maker);
        swap(_deleter, other._deleter);
    }

    void reset()
    {
        indexed_tl_ptr{}.swap(*this);
    }

    friend bool operator==(indexed_tl_ptr const& rhs, indexed_tl_ptr const& lhs) noexcept
    {
        return rhs._instance_idx && (rhs._instance_idx == lhs._instance_idx);
    }

    friend bool operator!=(indexed_tl_ptr const& rhs, indexed_tl_ptr const& lhs) noexcept
    {
        return !(rhs == lhs);
    }
};
