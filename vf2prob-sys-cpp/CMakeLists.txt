# VF2Prob C++ impl
# jhlade

cmake_minimum_required(VERSION 3.20)
project(vf2prob LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(VF2PROB_BUILD_TESTS "Build unit/property tests" ON)
option(VF2PROB_WITH_CURL   "Link libcurl for SNAP download (else shell out to curl)" OFF)
option(VF2PROB_WITH_ZLIB   "Link zlib for .gz decompression (else shell out to gunzip)" OFF)
# Opt-in -march=native; OFF by default so the released build stays portable.
option(VF2PROB_NATIVE      "Tune for the build host (-march=native)" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    # full check
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(VF2PROB_NATIVE AND NOT MSVC)
    add_compile_options(-march=native)
endif()

# Core library
add_library(vf2prob_core
    src/graph.cpp
    src/compat.cpp
    src/ordering.cpp
    src/assignment.cpp
    src/bounds.cpp
    src/algorithms/vf2.cpp
    src/algorithms/vf2pp.cpp
    src/algorithms/vf2bin.cpp
    src/algorithms/vf2prob.cpp
    src/algorithms/mpm.cpp
    src/io/graphml.cpp
    src/io/queries.cpp
    src/io/csv.cpp
    src/data/synthetic.cpp
    src/ml/train.cpp
    src/ml/calibrate.cpp
    src/ml/ranker.cpp
    src/registry.cpp
)
target_include_directories(vf2prob_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(vf2prob_core PUBLIC cxx_std_20)

if(VF2PROB_WITH_CURL)
    find_package(CURL REQUIRED)
    target_link_libraries(vf2prob_core PUBLIC CURL::libcurl)
    target_compile_definitions(vf2prob_core PUBLIC VF2PROB_HAVE_CURL=1)
endif()
if(VF2PROB_WITH_ZLIB)
    find_package(ZLIB REQUIRED)
    target_link_libraries(vf2prob_core PUBLIC ZLIB::ZLIB)
    target_compile_definitions(vf2prob_core PUBLIC VF2PROB_HAVE_ZLIB=1)
endif()

# Applications
add_executable(vf2prob_run apps/vf2prob_run.cpp)
target_link_libraries(vf2prob_run PRIVATE vf2prob_core)

add_executable(aggregate apps/aggregate.cpp)
target_link_libraries(aggregate PRIVATE vf2prob_core)

add_executable(train_weights apps/train_weights.cpp)
target_link_libraries(train_weights PRIVATE vf2prob_core)

add_executable(recovery apps/recovery.cpp)
target_link_libraries(recovery PRIVATE vf2prob_core)

add_executable(ranker_demo apps/ranker_demo.cpp)
target_link_libraries(ranker_demo PRIVATE vf2prob_core)

# Tests
if(VF2PROB_BUILD_TESTS)
    enable_testing()
    add_executable(vf2prob_tests
        tests/test_main.cpp
        tests/test_compat.cpp
        tests/test_assignment.cpp
        tests/test_bounds_admissible.cpp
        tests/test_optimum_invariant.cpp
        tests/test_continuous_kernel.cpp
        tests/test_io_roundtrip.cpp
        tests/test_vf2_baselines.cpp
        tests/test_ml.cpp
        tests/test_mpm.cpp
    )
    target_link_libraries(vf2prob_tests PRIVATE vf2prob_core)
    target_include_directories(vf2prob_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME vf2prob_tests COMMAND vf2prob_tests)
endif()
