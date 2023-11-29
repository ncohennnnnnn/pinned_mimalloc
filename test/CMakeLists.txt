set(TEST_SRCS test_arena)

# ------------------------------------------------------------------------------
# create exes for tests
# ------------------------------------------------------------------------------
foreach(test ${TEST_SRCS})
  add_executable(${test} ${test}.cpp)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test} PRIVATE pmimalloc)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
