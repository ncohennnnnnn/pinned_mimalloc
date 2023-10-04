find_package(PkgConfig QUIET)
pkg_check_modules(PC_NUMA QUIET numa)
set(NUMA_DEFINITIONS ${PC_NUMA_CFLAGS_OTHER})

find_path(NUMA_INCLUDE_DIR
    NAMES
        numa.h
        numaif.h
    HINTS
        ${NUMA_ROOT} ENV NUMA_ROOT
        ${PC_NUMA_INCLUDEDIR}
        ${PC_NUMA_INCLUDE_DIRS}
    PATH_SUFFIXES
        include)

find_library(NUMA_LIBRARY
    NAMES
        numa
        libnuma
    HINTS
        ${NUMA_ROOT} ENV NUMA_ROOT
        ${PC_NUMA_LIBDIR}
        ${PC_NUMA_LIBRARY_DIRS}
    PATH_SUFFIXES
        lib
        lib64)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

mark_as_advanced(NUMA_ROOT NUMA_INCLUDE_DIR NUMA_LIBRARY)

set(NUMA_LIBRARIES ${NUMA_LIBRARY} CACHE INTERNAL "")
set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR} CACHE INTERNAL "")

if(NOT TARGET NUMA::libnuma AND NUMA_FOUND)
    add_library(NUMA::libnuma SHARED IMPORTED)
    set_target_properties(NUMA::libnuma PROPERTIES
        IMPORTED_LOCATION ${NUMA_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${NUMA_INCLUDE_DIR})
endif()
