cmake_minimum_required(VERSION 3.22.1)
project(cuda_test CUDA CXX)

find_package(GTest REQUIRED)
find_package(OpenCV 4.0 REQUIRED)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

file(GLOB CXX_SRCS *.cpp)
file(GLOB CUDA_SRCS *.cu cuda_src/*.cu)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++20 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

add_library(test_cuda_lib ${CUDA_SRCS})
set_target_properties(test_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test ${CXX_SRCS})

target_compile_definitions(test PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data" OUTPUT_DIR="${CMAKE_BINARY_DIR}")
target_include_directories(test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
    test
    test_cuda_lib
    ${CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${GTEST_LIBRARIES} 
    ${GTEST_MAIN_LIBRARIES})
