cmake_minimum_required(VERSION 3.14)
project(pinned_mimalloc)

# ------------------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------------------
cmake_policy(SET CMP0028 NEW) # Double colon in target name means ALIAS or
# IMPORTED target
cmake_policy(SET CMP0060 NEW) # Link libraries by full path even in implicit
# directories
cmake_policy(SET CMP0074 NEW) # find_package uses XXX_ROOT vars

# ------------------------------------------------------------------------------
# Enable testing
# ------------------------------------------------------------------------------
include(CTest)
enable_testing()

# ------------------------------------------------------------------------------
# Set C++ standard to C++17
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Set the extra heap size, device option
# ------------------------------------------------------------------------------
set( EXT_HEAP_SZ_EXP 25 CACHE STRING "Size of the extra mimalloc heap : 1 << X, max = 29" )
option( ENABLE_DEVICE OFF)

# ------------------------------------------------------------------------------
# Cmake module path for find_package
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
# Setup Output directories (all binaries in bin/ change libs to lib/ if you
# want)
# ------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
${PROJECT_BINARY_DIR}/bin
CACHE PATH "Single Directory for all Executables."
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
${PROJECT_BINARY_DIR}/bin
CACHE PATH "Single Directory for all Libraries"
)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
${PROJECT_BINARY_DIR}/bin
CACHE PATH "Single Directory for all static libraries."
)

# # Activating PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_POSITION_INDEPENDENT_CODE)
    message(STATUS "PIE is supported.")
    # Add PIE-related flags or configuration here if needed.
else()
    message(STATUS "PIE is not supported.")
    # Handle the case where PIE is not supported.
endif()
# include(CheckPIESupported)
# check_pie_supported()
# set_property(TARGET pinned_mimalloc PROPERTY POSITION_INDEPENDENT_CODE TRUE)
# if(NOT CMAKE_C_LINK_PIE_SUPPORTED)
#   message(WARNING "PIE is not supported at link time: ${output}.\n"
#                   "PIE link options will not be passed to linker.")
# endif()


# Create config file
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config.hpp.in ${PROJECT_SOURCE_DIR}/include/config.hpp
)

# Finding mimalloc
# find_package(mimalloc REQUIRED)
# message(STATUS "Found mimalloc: ${MIMALLOC_FOUND}")

# Fetching mimalloc
# if(NOT _Mimalloc_already_fetched)
    #     set(_Mimalloc_repository "https://github.com/microsoft/mimalloc.git")
    #     set(_Mimalloc_tag "master")
    #     message(STATUS "Fetching MIMALLOC ${_Mimalloc_tag} from ${_Mimalloc_repository}")

    #   include(FetchContent)
    #   fetchcontent_declare(
    #     Mimalloc
    #     GIT_REPOSITORY ${_Mimalloc_repository}
    #     GIT_TAG ${_Mimalloc_tag}
    #   )
    #   fetchcontent_populate(Mimalloc)

    #   set(_Mimalloc_already_fetched ON CACHE INTERNAL "")
# endif()


# Finding fmt
find_package(fmt REQUIRED)
message(STATUS "Found fmt: ${FMT_FOUND}")

# Add the executable target
add_executable( pinned_mimalloc src/test.cpp )
# add_subdirectory(${Mimalloc_SOURCE_DIR} ${Mimalloc_BINARY_DIR})
add_subdirectory(extern/mimalloc)
target_compile_definitions(pinned_mimalloc PRIVATE ENABLE_DEVICE)
target_include_directories( pinned_mimalloc PUBLIC src include )
target_link_libraries( pinned_mimalloc PUBLIC mimalloc PRIVATE fmt::fmt-header-only )

# create a `ctest` test that executes the pinned allocator note that ctest
# doesn't use a shell, so we can't redirect the output to a file using the
# usual '>' operator
add_test(NAME pinned_mimalloc
         COMMAND "$<TARGET_FILE:pinned_mimalloc>"
)
