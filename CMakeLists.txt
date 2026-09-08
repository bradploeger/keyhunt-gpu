cmake_minimum_required(VERSION 3.18)
project(keyhunt_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Compute capability. Override with -DCMAKE_CUDA_ARCHITECTURES=86
# (75 Turing, 86 Ampere, 89 Ada, 90 Hopper). "native" needs CMake 3.24+.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    message(FATAL_ERROR
      "Set the GPU architecture explicitly, e.g. -DCMAKE_CUDA_ARCHITECTURES=86")
  endif()
endif()

set(HSIZE  128 CACHE STRING "half group size; 2*HSIZE keys per batch inversion")
set(FILTER 18  CACHE STRING "log2 of prefilter bits; 18 = 32 KB shared per block")

set(KH_DEFS HSIZE=${HSIZE} FILTER_LOG2_BITS=${FILTER})
if(MSVC)
  # hex256 uses sprintf and load_targets uses fopen; MSVC deprecates both
  list(APPEND KH_DEFS _CRT_SECURE_NO_WARNINGS)
endif()

add_executable(keyhunt-gpu src/search.cu)
target_compile_definitions(keyhunt-gpu PRIVATE ${KH_DEFS})
target_compile_options(keyhunt-gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas -O3,-v>)

# ---- host-side tests, no GPU required
enable_testing()
foreach(t test_math test_kernel_logic test_targets test_progress)
  add_executable(${t} test/${t}.cpp)
  target_compile_definitions(${t} PRIVATE ${KH_DEFS})
  add_test(NAME ${t} COMMAND ${t})
endforeach()

message(STATUS "architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "HSIZE=${HSIZE}  FILTER=${FILTER}")
