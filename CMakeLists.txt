cmake_minimum_required(VERSION 3.14)
project(pmimalloc)

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
set(CMAKE_CXX_STANDARD 20)

# ------------------------------------------------------------------------------
# Set PIC
# ------------------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# Set the external heap size, device and logging option
# ------------------------------------------------------------------------------
# set( EXT_HEAP_SZ_EXP 25 CACHE STRING "Size of the extra mimalloc heap : 1 << X with 24 < X < 31" )
# if (${EXT_HEAP_SZ_EXP} LESS 25 OR ${EXT_HEAP_SZ_EXP} GREATER 31)
#     message(FATAL_ERROR "Invalid value for EXT_HEAP_SZ_EXP. Value must be within the range [25,31]")
# endif()
# option( PMIMALLOC_ENABLE_DEVICE OFF)
# option( PMIMALLOC_ENABLE_LOGGING OFF)

# ------------------------------------------------------------------------------
# Set the compiler options
# ------------------------------------------------------------------------------
option( WITH_MIMALLOC ON )

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

find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA : ${CUDA_FOUND}")

find_package(HWLOC REQUIRED)
message(STATUS "Found HWLOC : ${HWLOC_FOUND}")

# ---------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
set(ENABLE_LOGGING OFF CACHE BOOL "print logging info to cerr")

# ---------------------------------------------------------------------
# Generate config file
# ---------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp @ONLY)

# ---------------------------------------------------------------------
# Logging and mimalloc options
# ---------------------------------------------------------------------
if(ENABLE_LOGGING)
    set(PMI_LOG_SRC src/log.cpp)
endif()

# ------------------------------------------------------------------------------
# Building executable target
# ------------------------------------------------------------------------------
add_executable( pmimalloc
            src/test.cpp
            src/thread_affinity.cpp
            src/task_group.cpp
            src/numa.cpp
            ${PMI_LOG_SRC}
            )
target_include_directories( pmimalloc PUBLIC include )
target_link_libraries( pmimalloc
                    # PUBLIC
                    #     mimalloc
                    PRIVATE
                        fmt::fmt-header-only
                        NUMA::libnuma
                        CUDA::cudart
                        HWLOC::libhwloc
                    )
target_compile_definitions(pmimalloc PRIVATE WITH_MIMALLOC )


if (WITH_MIMALLOC)
    include(external)
    set(MI_OVERRIDE OFF)
    get_external_project(
      PROJECT_NAME "mimalloc"
      GIT_REPO "https://github.com/microsoft/mimalloc"
      GIT_TAG "master"
      PREFER_LOCAL
    )

    # target_compile_definitions(pmimalloc PRIVATE -DMI_OVERRIDE=OFF )
    target_link_libraries( pmimalloc PUBLIC mimalloc )
    target_sources(pmimalloc PRIVATE src/ex_mimalloc.cpp)
endif()

# ---------------------------------------------------------------------
# NUMA setup
# ---------------------------------------------------------------------
if(UNIX AND NOT APPLE)
find_package(NUMA REQUIRED)
target_link_libraries(pmimalloc PRIVATE NUMA::libnuma)
set(NUMA_THROWS OFF CACHE BOOL "True if numa_tools may throw during initialization")
if (NUMA_THROWS)
    target_compile_definitions(pmimalloc PRIVATE NUMA_THROWS)
endif()
set(NUMA_FOR_LOCAL OFF CACHE BOOL "Use numa_tools for local node allocations")
if (NUMA_FOR_LOCAL)
    target_compile_definitions(pmimalloc PRIVATE NUMA_FOR_LOCAL)
endif()
else()
message(WARNING "${CMAKE_PROJECT_NAME} configured without NUMA support on Mac")
endif()

# ------------------------------------------------------------------------------
# Choice of backend provider
# ------------------------------------------------------------------------------
set(BACKEND "none" CACHE STRING "Choose the type of the backend.")
set_property(CACHE BACKEND PROPERTY STRINGS "libfabric" "ucx" "mpi" "none")

string(TOUPPER ${BACKEND} BACKEND_U)
set(BACKEND_ ${BACKEND_U} CACHE INTERNAL "" FORCE)
# set(PMIMALLOC_DEVICE "PMIMALLOC_DEVICE_${BACKEND_}" CACHE INTERNAL "" FORCE)

include(GNUInstallDirs)

if (${BACKEND} STREQUAL "libfabric")
    set( BACKEND_SOURCE src/${BACKEND}/backend.cpp )
    find_package(Libfabric REQUIRED)
    message(STATUS "Found Libfabric : ${LIBFABRIC_FOUND}")
    find_package(Boost REQUIRED)
    message(STATUS "Found Boost : ${BOOST_FOUND}")
    target_link_libraries(pmimalloc
                        PRIVATE
                            libfabric::libfabric
                            Boost::boost
                        )
    set( WITH_LIBFABRIC ON )
    target_compile_definitions( pmimalloc PRIVATE WITH_LIBFABRIC )
    target_sources(pmimalloc PRIVATE ${BACKEND_SOURCE})
    target_include_directories( pmimalloc PUBLIC src/${BACKEND} )
# building the library
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp src/numa.cpp src/log.cpp)
    target_compile_definitions(pmimalloc_${BACKEND} PUBLIC WITH_MIMALLOC )
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.h)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include src/${BACKEND} ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND}
                        fmt::fmt-header-only
                        NUMA::libnuma
                        CUDA::cudart
                        libfabric::libfabric
                        Boost::boost
                        mimalloc
    )
    install(TARGETS pmimalloc_${BACKEND}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_BINARY_DIR}/pmimalloc_${BACKEND}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
elseif (${BACKEND} STREQUAL "ucx")
    set( BACKEND_SOURCE src/${BACKEND}/backend.cpp )
    find_package(UCX REQUIRED)
    message(STATUS "Found UCX : ${UCX_FOUND}")
    target_link_libraries(pmimalloc PRIVATE UCX::ucx)
    set( WITH_UCX ON )
    target_compile_definitions( pmimalloc PRIVATE WITH_UCX )
    target_sources(pmimalloc PRIVATE ${BACKEND_SOURCE})
    target_include_directories( pmimalloc PUBLIC src/${BACKEND} )
# building the library
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp src/numa.cpp src/log.cpp)
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.h)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include src/${BACKEND} ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND}
                        fmt::fmt-header-only
                        NUMA::libnuma
                        CUDA::cudart
                        UCX::ucx
    )
    install(TARGETS pmimalloc_${BACKEND}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_BINARY_DIR}/pmimalloc_${BACKEND}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
elseif (${BACKEND} STREQUAL "mpi")
    set( BACKEND_SOURCE src/${BACKEND}/backend.cpp )
    find_package(MPI REQUIRED)
    message(STATUS "Found MPI : ${MPI_FOUND}")
    target_link_libraries(pmimalloc PRIVATE MPI::MPI_CXX )
    set( WITH_MPI ON )
    target_compile_definitions( pmimalloc PRIVATE WITH_MPI )
    target_sources(pmimalloc PRIVATE ${BACKEND_SOURCE})
    target_include_directories( pmimalloc PUBLIC src/${BACKEND} )
# building the library
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp src/numa.cpp src/log.cpp)
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.h)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include src/${BACKEND}. ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND}
                        fmt::fmt-header-only
                        NUMA::libnuma
                        CUDA::cudart
                        MPI::MPI_CXX
    )
    install(TARGETS pmimalloc_${BACKEND}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_BINARY_DIR}/pmimalloc_${BACKEND}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
elseif (${BACKEND} STREQUAL "none")
    message("No backend selected.")
# building the library
    add_library(pmimalloc_${BACKEND}
            SHARED
                src/ex_mimalloc.cpp
                src/numa.cpp
                src/log.cpp
                src/thread_affinity.cpp
                src/task_group.cpp
                )
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.hpp)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND}
                        fmt::fmt-header-only
                        NUMA::libnuma
                        CUDA::cudart
                        HWLOC::libhwloc
    )
    install(TARGETS pmimalloc_${BACKEND}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    # install(FILES ${CMAKE_BINARY_DIR}/pmimalloc_${BACKEND}.pc
    #     DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
endif()

add_library(pmimalloc_target ALIAS pmimalloc_${BACKEND})

# ------------------------------------------------------------------------------
# CTESTs
# ------------------------------------------------------------------------------
# create a `ctest` test that executes the pinned allocator note that ctest
# doesn't use a shell, so we can't redirect the output to a file using the
# usual '>' operator
add_test(NAME pmimalloc
         COMMAND "$<TARGET_FILE:pmimalloc>"
)
