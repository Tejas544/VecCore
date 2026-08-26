include(FetchContent)

# doctest is a single header and a tiny repo.  D6 permits standard tooling:
# writing our own test framework would demonstrate nothing that using one and
# explaining it does not.
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(veccore_tests
  test_smoke.cpp
  test_phase1.cpp
  test_hnsw.cpp
  test_pq.cpp
  test_bm25.cpp)

target_link_libraries(veccore_tests PRIVATE veccore doctest::doctest)

include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")
doctest_discover_tests(veccore_tests)
