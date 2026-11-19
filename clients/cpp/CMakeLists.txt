cmake_minimum_required(VERSION 3.20)
project(gpurent_worker LANGUAGES CXX)

option(GPURENT_CUDA "Enable the ONNX Runtime CUDA execution provider" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an ONNX Runtime C++ SDK installation")
endif()

include(FetchContent)
FetchContent_Declare(ixwebsocket
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.6
  GIT_SHALLOW TRUE
)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
set(USE_TLS OFF CACHE BOOL "" FORCE)
set(USE_ZLIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ixwebsocket)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(gpurent-worker main.cpp)
target_include_directories(gpurent-worker PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_directories(gpurent-worker PRIVATE "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(gpurent-worker PRIVATE ixwebsocket onnxruntime nlohmann_json::nlohmann_json)
if(GPURENT_CUDA)
  target_compile_definitions(gpurent-worker PRIVATE GPURENT_CUDA)
endif()
