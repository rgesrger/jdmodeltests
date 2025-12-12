cmake_minimum_required(VERSION 3.14)
project(distilbert_infer CXX)

# Allow using the vendored ONNX Runtime if system package isn't installed.
if(NOT DEFINED onnxruntime_DIR)
	set(_onnxruntime_guess "${CMAKE_CURRENT_LIST_DIR}/onnxruntime-linux-x64-1.23.2/lib/cmake/onnxruntime")
	if(EXISTS "${_onnxruntime_guess}/onnxruntimeConfig.cmake")
		set(onnxruntime_DIR "${_onnxruntime_guess}" CACHE PATH "Path to onnxruntimeConfig.cmake")
	endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

find_package(onnxruntime REQUIRED)

add_executable(distilbert_infer distilbert_infer.cpp)
add_executable(distilbert_service distilbert_service.cpp)
add_executable(gateway gateway.cpp ${CMAKE_CURRENT_LIST_DIR}/../../faasd/junctiond/junctiond.cpp)

target_link_libraries(distilbert_infer PRIVATE onnxruntime::onnxruntime Threads::Threads)
target_link_libraries(distilbert_service PRIVATE onnxruntime::onnxruntime Threads::Threads)
target_link_libraries(gateway PRIVATE onnxruntime::onnxruntime Threads::Threads)

# Add junctiond headers (from faasd/junctiond) so gateway can call JunctionD directly.
target_include_directories(gateway PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../faasd/junctiond)

target_include_directories(gateway PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../junctiond)
