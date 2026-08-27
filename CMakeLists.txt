cmake_minimum_required(VERSION 3.22)
project(veccore VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options.  See PLAN.md Phase 0.3 and CONTEXT.md D2 / D10.
# ---------------------------------------------------------------------------
option(VECCORE_BUILD_TESTS "Build the doctest suite"                    ON)
option(VECCORE_BUILD_BENCH "Build the benchmark harness"                ON)
option(VECCORE_BUILD_TOOLS "Build the developer tools (hybrid_eval)"    ON)
option(VECCORE_ASAN        "AddressSanitizer + UBSan (Debug default)"   OFF)
option(VECCORE_TSAN        "ThreadSanitizer -- Phase 5, separate build" OFF)
option(VECCORE_NATIVE      "-march=native in Release"                   ON)
option(VECCORE_VEC_REPORT  "Emit the auto-vectorisation report (D11)"   OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# pybind11 in Phase 6 links this static lib into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# All executables in one predictable place.  Not the build root: the `bench`
# executable would collide with the `bench/` subdirectory CMake creates for that
# subproject, and the link fails with an unhelpful "ld returned 1".
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Debug implies sanitizers.  L-01: this is the entire reason the project builds
# on Linux rather than on the local MSYS2 toolchain, so it is on by default and
# has to be switched off deliberately.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT VECCORE_TSAN)
  set(VECCORE_ASAN ON CACHE BOOL "AddressSanitizer + UBSan" FORCE)
endif()

if(VECCORE_ASAN AND VECCORE_TSAN)
  message(FATAL_ERROR
    "ASan and TSan cannot be combined.  Use two build directories:\n"
    "  cmake -B ~/veccore-build-debug -DCMAKE_BUILD_TYPE=Debug\n"
    "  cmake -B ~/veccore-build-tsan  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVECCORE_TSAN=ON")
endif()

# ---------------------------------------------------------------------------
# Flags.
#
# P-02: Release is -O3 and NOT -Ofast/-ffast-math.  Fast-math reassociates
# floating point sums, so distances stop matching the ground truth bit-for-bit
# and a compiler flag masquerades as a recall bug.  If you ever want it, make it
# a separate build and a separate row in the results table.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3 -fno-omit-frame-pointer")

add_library(veccore_flags INTERFACE)
target_compile_options(veccore_flags INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wdouble-promotion)

set(VECCORE_SANITIZER_DESC "none")

if(VECCORE_NATIVE)
  target_compile_options(veccore_flags INTERFACE $<$<CONFIG:Release>:-march=native>)
  target_compile_options(veccore_flags INTERFACE $<$<CONFIG:RelWithDebInfo>:-march=native>)
endif()

if(VECCORE_VEC_REPORT)
  # D11: check the compiler actually vectorised the distance loop before
  # reaching for intrinsics.
  target_compile_options(veccore_flags INTERFACE -fopt-info-vec-optimized -fopt-info-vec-missed)
endif()

if(VECCORE_ASAN)
  # -fno-sanitize-recover=all makes UBSan abort instead of printing and
  # continuing.  A UBSan finding that scrolls past is a finding you did not get.
  target_compile_options(veccore_flags INTERFACE
    -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  target_link_options(veccore_flags INTERFACE -fsanitize=address,undefined)
  set(VECCORE_SANITIZER_DESC "address,undefined")
endif()

if(VECCORE_TSAN)
  target_compile_options(veccore_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(veccore_flags INTERFACE -fsanitize=thread)
  set(VECCORE_SANITIZER_DESC "thread")
endif()

find_package(Threads REQUIRED)
target_link_libraries(veccore_flags INTERFACE Threads::Threads)

# ---------------------------------------------------------------------------
# Build stamp (D10).  Every JSON record carries these, so a number in the README
# can always be traced back to the build that produced it.
# ---------------------------------------------------------------------------
# CMake's per-config flag variables are UPPERCASE (CMAKE_CXX_FLAGS_DEBUG), while
# CMAKE_BUILD_TYPE keeps whatever case you typed.  Without TOUPPER this expands
# to CMAKE_CXX_FLAGS_Debug, which does not exist, and the stamp records an empty
# flags string -- a D10 record that looks complete and says nothing.
string(TOUPPER "${CMAKE_BUILD_TYPE}" VECCORE_BUILD_TYPE_UC)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${VECCORE_BUILD_TYPE_UC}}" VECCORE_BASE_FLAGS)
set(VECCORE_EFFECTIVE_FLAGS "${VECCORE_BASE_FLAGS}")
if(VECCORE_NATIVE AND (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
  set(VECCORE_EFFECTIVE_FLAGS "${VECCORE_EFFECTIVE_FLAGS} -march=native")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/veccore/build_info.hpp"
  @ONLY)

# ---------------------------------------------------------------------------
# The library.  Knows nothing about pybind11, JSON transport, or HTTP -- those
# are separate targets that depend on it, never the reverse (PLAN.md section 3).
# ---------------------------------------------------------------------------
add_library(veccore STATIC
  src/stamp.cpp
  src/xvecs.cpp
  src/serialize.cpp
  src/hnsw.cpp
  src/kmeans.cpp
  src/pq.cpp
  src/bm25.cpp
  src/fusion.cpp)

target_include_directories(veccore PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_link_libraries(veccore PUBLIC veccore_flags)

# ---------------------------------------------------------------------------
# Phase 0 gate: this must abort with a heap-buffer-overflow report.
# If it exits cleanly, the sanitizers are not actually running and every
# assumption PLAN.md makes about catching memory bugs is false (L-01).
# ---------------------------------------------------------------------------
if(VECCORE_ASAN)
  add_executable(asan_probe tools/asan_probe.cpp)
  target_link_libraries(asan_probe PRIVATE veccore)
endif()

# Phase 5 gate: this must produce a ThreadSanitizer data-race report.
# A clean TSan run over the test suite proves nothing unless TSan can be shown
# to fire on this build -- L-01's lesson, applied to a second sanitizer.
if(VECCORE_TSAN)
  add_executable(tsan_probe tools/tsan_probe.cpp)
  target_link_libraries(tsan_probe PRIVATE veccore)
endif()

# EdgeRAG hybrid evaluation (D7). Lives in tools/ because the TF-IDF baseline
# it contains is a baseline, not a component -- same treatment as FAISS.
#
# Gated so a wheel build does not compile it: `pip install .` wants the extension
# and nothing else, and an evaluation harness is a developer target.
if(VECCORE_BUILD_TOOLS)
  add_executable(hybrid_eval tools/hybrid_eval.cpp)
  target_link_libraries(hybrid_eval PRIVATE veccore)
endif()

# ---------------------------------------------------------------------------
# Python bindings. Optional: the library must build and test without them, so a
# missing pybind11 is a skipped target and not a broken build.
# ---------------------------------------------------------------------------
option(VECCORE_BUILD_PYTHON "Build the pybind11 module" ON)
if(VECCORE_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_veccore bindings/veccore_py.cpp)
    target_link_libraries(_veccore PRIVATE veccore)
    message(STATUS "  python     : pybind11 ${pybind11_VERSION} -> _veccore")

    # Without this the wheel contains the two .py files and no extension, and
    # `pip install veccore` produces a package whose import fails. scikit-build-core
    # packages what CMake's install() rules emit, and pybind11_add_module does not
    # add one -- a 19 KB wheel is the only symptom, which is easy to miss because
    # the build itself succeeds.
    if(SKBUILD)
      install(TARGETS _veccore DESTINATION veccore)
    endif()
  else()
    message(STATUS "  python     : pybind11 not found, skipping _veccore")
  endif()
endif()

if(VECCORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VECCORE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

message(STATUS "veccore ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  flags      : ${VECCORE_EFFECTIVE_FLAGS}")
message(STATUS "  sanitizers : ${VECCORE_SANITIZER_DESC}")
