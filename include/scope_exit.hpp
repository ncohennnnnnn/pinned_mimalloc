#pragma once

#include <type_traits>
#include <utility>


template<typename F>
class scope_exit {
  public:
    template<typename F2,
        typename = std::enable_if_t<std::is_nothrow_constructible<F, F2&&>::value>>
    scope_exit(F2&& f) noexcept : on_exit{std::forward<F2>(f)} {}
    scope_exit(scope_exit const&) = delete;
    scope_exit(scope_exit&&) = delete;
    ~scope_exit() noexcept(noexcept(on_exit())) { on_exit(); }

  private:
    F on_exit;
};

template<typename F>
auto on_scope_exit(F&& f) {
    return scope_exit<std::decay_t<F>>{std::forward<F>(f)};
}

