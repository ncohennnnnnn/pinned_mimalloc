set(TEST_SRCS
    test_host
    test_host_threads
    test_host_arenas
    test_host_arenas_threads
    test_mirror
    test_mirror_threads
    test_mirror_arenas
    test_mirror_arenas_threads
    test_host_pmr
    test_host_pmr_threads
)

# ------------------------------------------------------------------------------
# create exes for tests
# ------------------------------------------------------------------------------
foreach(test ${TEST_SRCS})
  add_executable(${test} ${test}.cpp)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test} PRIVATE pmimalloc)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
