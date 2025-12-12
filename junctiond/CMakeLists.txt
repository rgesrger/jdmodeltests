cmake_minimum_required(VERSION 3.15)
project(junctiond)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Find Dependencies ---
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

message(STATUS "Using Protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

# --- 2. Setup Paths ---
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GEN_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/gen")
set(PROTO_FILE "${PROTO_DIR}/junctiond.proto")

file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_SRCS
    "${GEN_DIR}/junctiond.pb.cc"
    "${GEN_DIR}/junctiond.grpc.pb.cc"
)
set(PROTO_HDRS
    "${GEN_DIR}/junctiond.pb.h"
    "${GEN_DIR}/junctiond.grpc.pb.h"
)

# --- 3. Generate Codes ---
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --grpc_out=${GEN_DIR}
         --cpp_out=${GEN_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         -I "${PROTO_DIR}"
         "${PROTO_FILE}"
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC C++ code..."
    VERBATIM
)

# --- 4. Include Paths (THE FIX) ---
# We explicitly add /usr/local/include to ensure standard headers are found
include_directories(
    SYSTEM "/usr/local/include"
    "${GEN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

# --- 5. Build ---
add_executable(junctiond
    junctiond_server.cpp
    junctiond.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

# Link against the targets (Config mode uses protobuf::libprotobuf)
target_link_libraries(junctiond
    PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
)