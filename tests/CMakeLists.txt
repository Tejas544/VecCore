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
  test_bm25.cpp
  test_concurrent.cpp
  test_serialize.cpp)

target_link_libraries(veccore_tests PRIVATE veccore doctest::doctest)

# L-09, and the half of it the README could not fix by documenting a command.
#
# TSan maps shadow memory at fixed addresses that this kernel's ASLR range
# overlaps, so a TSan binary aborts at startup nondeterministically unless it
# runs under `setarch -R`. `setarch -R ctest` covers the test *run* -- but
# doctest_discover_tests also executes the binary at BUILD time to enumerate the
# test cases, and nothing the user types can wrap that. The symptom is a build
# that fails with "FATAL: ThreadSanitizer: unexpected memory mapping" and an
# empty error message from CMake, which looks like a compile error and is not.
#
# CROSSCOMPILING_EMULATOR is the launcher doctest threads through to both the
# enumeration command and every generated add_test(), so setting it here fixes
# the build and makes a bare `ctest` correct as well.
if(VECCORE_TSAN)
  set_target_properties(veccore_tests PROPERTIES CROSSCOMPILING_EMULATOR "setarch;-R")
endif()

include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")
doctest_discover_tests(veccore_tests)
