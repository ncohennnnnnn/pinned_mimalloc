#pragma once

#include <fmt/core.h>

#include <pmimalloc/config.hpp>

// use the variadic macro LOG to log messages
// the arguments will be divided by a whitespace
#ifdef ENABLE_LOGGING
# define LOG(...) ::log_message(__VA_ARGS__, "(", __FILE__, ":", __LINE__, ")");
#else
# define LOG(...)
#endif

// implementation
#ifdef ENABLE_LOGGING
# include <sstream>

std::stringstream& log_stream();

void print_log_message(std::stringstream&);

void log_message(std::stringstream&);

template <typename S, typename... Rest>
void log_message(std::stringstream& str, S&& s, Rest&&... r)
{
    str << " " << s;
    log_message(str, std::forward<Rest>(r)...);
}

// main logging function
template <typename... S>
void log_message(S&&... s)
{
    auto& str = log_stream();
    log_message(str, std::forward<S>(s)...);
    print_log_message(str);
}
#endif