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
# Set PIC
# ------------------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# Set the external heap size, device and logging option
# ------------------------------------------------------------------------------
set( EXT_HEAP_SZ_EXP 25 CACHE STRING "Size of the extra mimalloc heap : 1 << X with 24 < X < 31" )
if (${EXT_HEAP_SZ_EXP} LESS 25 OR ${EXT_HEAP_SZ_EXP} GREATER 31)
    message(FATAL_ERROR "Invalid value for EXT_HEAP_SZ_EXP. Value must be within the range [25,31]")
endif()

option( PMIMALLOC_ENABLE_DEVICE OFF)
option( PMIMALLOC_ENABLE_LOGGING OFF)

# ------------------------------------------------------------------------------
# Create config file
# ------------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config.hpp.in 
    ${PROJECT_SOURCE_DIR}/include/config.hpp
)

# ------------------------------------------------------------------------------
# Cmake module path for find_package
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
# Setup Output directories
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

# ------------------------------------------------------------------------------
# Finding libraries
# ------------------------------------------------------------------------------
find_package(fmt REQUIRED)
message(STATUS "Found fmt: ${FMT_FOUND}")

find_package(NUMA REQUIRED)
message(STATUS "Found NUMA: ${NUMA_FOUND}")

# ------------------------------------------------------------------------------
# Building executable target
# ------------------------------------------------------------------------------
add_executable( pinned_mimalloc src/test.cpp src/mimalloc.cpp )
add_subdirectory(extern/mimalloc)
target_compile_definitions(pinned_mimalloc PRIVATE PMIMALLOC_ENABLE_DEVICE PMIMALLOC_ENABLE_LOGGING)
target_include_directories( pinned_mimalloc PUBLIC src include )
target_link_libraries( pinned_mimalloc 
                    PUBLIC mimalloc
                    PRIVATE fmt::fmt-header-only NUMA::libnuma 
)

# ------------------------------------------------------------------------------
# Choice of GPU language
# ------------------------------------------------------------------------------
if (PMIMALLOC_ENABLE_LOGGING)
    target_sources(pinned_mimalloc PRIVATE src/log.cpp)
endif()

if (PMIMALLOC_ENABLE_DEVICE)
    set(PMIMALLOC_DEVICE_RUNTIME "cuda" CACHE STRING "Choose the type of the gpu runtime.")
    set_property(CACHE PMIMALLOC_DEVICE_RUNTIME PROPERTY STRINGS "cuda" "hip" "emulate")

    string(TOUPPER ${PMIMALLOC_DEVICE_RUNTIME} PMIMALLOC_DEVICE_RUNTIME_U)
    set(PMIMALLOC_DEVICE_RUNTIME_ ${PMIMALLOC_DEVICE_RUNTIME_U} CACHE INTERNAL "" FORCE)
    set(PMIMALLOC_DEVICE "PMIMALLOC_DEVICE_${PMIMALLOC_DEVICE_RUNTIME_}" CACHE INTERNAL "" FORCE)

    if (${PMIMALLOC_DEVICE_RUNTIME} STREQUAL "cuda")
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "Found CUDA: ${CUDA_FOUND}")
        target_link_libraries(pinned_mimalloc PRIVATE CUDA::cudart)
    elseif (${PMIMALLOC_DEVICE_RUNTIME} STREQUAL "hip")
        find_package(hip REQUIRED)
        message(STATUS "Found hip: ${HIP_FOUND}")
        target_link_libraries(pinned_mimalloc PRIVATE hip::device)
        # workaround for clang-hip
        target_compile_options(pinned_mimalloc PRIVATE $<BUILD_INTERFACE:-std=c++14>)
    endif()

    target_sources(pinned_mimalloc PRIVATE src/device_${PMIMALLOC_DEVICE_RUNTIME}.cpp)
else()
    set(PMIMALLOC_DEVICE "PMIMALLOC_DEVICE_NONE" CACHE INTERNAL "" FORCE)
    target_sources(pinned_mimalloc PRIVATE src/device_none.cpp)
endif()

# ------------------------------------------------------------------------------
# CTESTs
# ------------------------------------------------------------------------------
# create a `ctest` test that executes the pinned allocator note that ctest
# doesn't use a shell, so we can't redirect the output to a file using the
# usual '>' operator
add_test(NAME pinned_mimalloc
         COMMAND "$<TARGET_FILE:pinned_mimalloc>"
)
