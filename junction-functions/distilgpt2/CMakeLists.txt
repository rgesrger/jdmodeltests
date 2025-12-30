cmake_minimum_required(VERSION 3.14)
project(distilgpt2_infer CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pthreads
find_package(Threads REQUIRED)

# -------------------------------
# ONNX Runtime paths
# -------------------------------
# Adjust this path to where you extracted onnxruntime-linux-x64-1.16.3
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# -------------------------------
# Executable
# -------------------------------
add_executable(distilgpt2_infer distilgpt2_infer.cpp)

# Link libraries
target_link_libraries(distilgpt2_infer
    onnxruntime
    Threads::Threads
)
