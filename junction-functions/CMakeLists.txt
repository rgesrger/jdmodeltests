cmake_minimum_required(VERSION 3.14)
project(distilbert_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime REQUIRED)

add_executable(distilbert_infer distilbert_infer.cpp)

target_link_libraries(distilbert_infer PRIVATE onnxruntime::onnxruntime)
