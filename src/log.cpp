#include <iostream>
#include <mutex>
#include <pmimalloc/log.hpp>
#include <sstream>

namespace {
    std::mutex& log_mutex()
    {
        static std::mutex m;
        return m;
    }
}    // namespace

std::stringstream& log_stream()
{
    constexpr char prefix[] = "PMIMALLOC:";
    static thread_local std::stringstream str;
    str.clear();
    str << prefix;
    return str;
}

/* std::cout version */
void print_log_message(std::stringstream& str)
{
    std::lock_guard<std::mutex> m(log_mutex());
    str << "\n";
    str >> std::cerr.rdbuf();
    std::cerr.flush();
}

/* fmt::print version */
// void print_log_message(std::stringstream& str)
// {
//     std::lock_guard<std::mutex> m(log_mutex());
//     fmt::print("{}\n", str.str());
// }

void log_message(std::stringstream&) {}