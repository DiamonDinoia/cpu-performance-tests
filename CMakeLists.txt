cmake_minimum_required(VERSION 3.16)
project(CPUTests VERSION 1.0 LANGUAGES C CXX)


set(CPM_DOWNLOAD_VERSION 0.36.0)
if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.7.1
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

CPMAddPackage(
  NAME xsimd
  GITHUB_REPOSITORY xtensor-stack/xsimd
  GIT_TAG 13.2.0
)

find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(latency-test latency.cpp)
target_link_libraries(latency-test benchmark)
target_compile_features(latency-test PRIVATE cxx_std_17) # This means at least C++17

add_executable(branch-test branch.cpp)
target_compile_features(branch-test PRIVATE cxx_std_17) # This means at least C++17

add_executable(memory-bandwidth bandwidth.cpp)
target_link_libraries(memory-bandwidth PRIVATE xsimd OpenMP::OpenMP_CXX)
target_compile_features(memory-bandwidth PRIVATE cxx_std_17)
target_compile_options(memory-bandwidth PRIVATE -march=native)
