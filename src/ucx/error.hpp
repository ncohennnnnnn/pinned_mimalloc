#pragma once

#include <ucp/api/ucp.h>

#ifdef NDEBUG
# define OOMPH_CHECK_UCX_RESULT(x) x;
# define OOMPH_CHECK_UCX_RESULT_NOEXCEPT(x) x;
#else
# include <stdexcept>
# include <string>
# define OOMPH_CHECK_UCX_RESULT(x)                                                                 \
     if (x != UCS_OK)                                                                              \
         throw std::runtime_error("OOMPH Error: UCX Call failed " + std::string(#x) + " in " +     \
             std::string(__FILE__) + ":" + std::to_string(__LINE__));
# define OOMPH_CHECK_UCX_RESULT_NOEXCEPT(x)                                                        \
     if (x != UCX_OK)                                                                              \
     {                                                                                             \
         std::cerr << "OOMPH Error: UCX Call failed " << std::string(#x) << " in "                 \
                   << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;       \
         std::terminate();                                                                         \
     }
#endif
