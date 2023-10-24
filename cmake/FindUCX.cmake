find_package(PkgConfig QUIET)
pkg_check_modules(PC_UCX QUIET ucx)

find_path(UCX_INCLUDE_DIR ucp/api/ucp.h
    HINTS
    ${HPCX_UCX_DIR}   ENV HPCX_UCX_DIR
    ${UCX_ROOT}  ENV UCX_ROOT
    ${UCX_DIR}   ENV UCX_DIR
    PATH_SUFFIXES include)

find_path(UCX_LIBRARY_DIR libucp.so
    HINTS
    ${HPCX_UCX_DIR}   ENV HPCX_UCX_DIR
    ${UCX_ROOT}  ENV UCX_ROOT
    ${UCX_DIR}   ENV UCX_DIR
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCX DEFAULT_MSG UCX_LIBRARY_DIR UCX_INCLUDE_DIR)
mark_as_advanced(UCX_LIBRARY_DIR UCX_INCLUDE_DIR)

if(NOT TARGET UCX::ucx AND UCX_FOUND)
  add_library(UCX::ucx INTERFACE IMPORTED)
  set_target_properties(UCX::ucx PROPERTIES
    INTERFACE_LINK_LIBRARIES "ucp;ucs;uct"
    INTERFACE_LINK_DIRECTORIES ${UCX_LIBRARY_DIR}
    INTERFACE_INCLUDE_DIRECTORIES ${UCX_INCLUDE_DIR}
  )
endif()
