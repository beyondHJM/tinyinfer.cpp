cmake_minimum_required(VERSION 3.18)
project(tinyinfer.cpp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(tinyinfer
    src/main.cpp
    src/gguf_reader.cpp
    src/tokenizer.cpp
    src/model.cpp
    src/selftest.cpp
    src/inference.cu
    src/cuda/rms_norm.cu
    src/cuda/rope.cu
    src/cuda/gemm.cu
    src/cuda/attention.cu
    src/cuda/ffn.cu
    src/cuda/lm_head.cu
)

target_link_libraries(tinyinfer PRIVATE CUDA::cudart)
target_include_directories(tinyinfer PRIVATE src src/cuda)
