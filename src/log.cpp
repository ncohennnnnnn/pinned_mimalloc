#include <log.hpp>

#include <iostream>
#include <sstream>
#include <mutex>


std::mutex&
log_mutex()
{
    static std::mutex m;
    return m;
}

std::stringstream&
log_stream()
{
    constexpr char                        prefix[] = "PMIMALLOC:";
    static thread_local std::stringstream str;
    str.clear();
    str << prefix;
    return str;
}

void
print_log_message(std::stringstream& str)
{
    std::lock_guard<std::mutex> m(log_mutex());
    str << "\n";
    str >> std::cerr.rdbuf();
    std::cerr.flush();
}

void
log_message(std::stringstream&)
{
}
