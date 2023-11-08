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
set(CMAKE_CXX_STANDARD 17)

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

# ------------------------------------------------------------------------------
# Building executable target
# ------------------------------------------------------------------------------
add_executable( pmimalloc src/test.cpp)
target_compile_definitions(pmimalloc 
                        PRIVATE 
                            WITH_MIMALLOC
                        )
target_include_directories( pmimalloc PUBLIC include )
target_link_libraries( pmimalloc 
                    PUBLIC 
                        mimalloc
                    PRIVATE 
                        fmt::fmt-header-only 
                        NUMA::libnuma 
                        CUDA::cudart
                    )
if (WITH_MIMALLOC)
    add_subdirectory(ext/mimalloc)
    target_sources(pmimalloc PRIVATE src/ex_mimalloc.cpp)
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
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp)
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.h)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include src/${BACKEND} ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND} 
                        fmt::fmt-header-only 
                        NUMA::libnuma 
                        CUDA::cudart
                        libfabric::libfabric
                        Boost::boost
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
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp)
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
    add_library(pmimalloc_${BACKEND} SHARED src/${BACKEND}/backend.cpp src/ex_mimalloc.cpp)
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
    add_library(pmimalloc_${BACKEND} SHARED src/ex_mimalloc.cpp)
    set_target_properties(pmimalloc_${BACKEND} PROPERTIES
        PUBLIC_HEADER include/allocator.h)
    target_include_directories(pmimalloc_${BACKEND} PUBLIC include ext/mimalloc/include)
    target_link_libraries(pmimalloc_${BACKEND} 
                        fmt::fmt-header-only 
                        NUMA::libnuma 
                        CUDA::cudart
    )
    install(TARGETS pmimalloc_${BACKEND}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_BINARY_DIR}/pmimalloc_${BACKEND}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
endif()


# ------------------------------------------------------------------------------
# CTESTs
# ------------------------------------------------------------------------------
# create a `ctest` test that executes the pinned allocator note that ctest
# doesn't use a shell, so we can't redirect the output to a file using the
# usual '>' operator
add_test(NAME pmimalloc
         COMMAND "$<TARGET_FILE:pmimalloc>"
)
