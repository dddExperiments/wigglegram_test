cmake_minimum_required(VERSION 3.16)
project(websiftgpu_cpp)

set(CMAKE_CXX_STANDARD 17)

# Output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Dependency: WebGPU (wgpu-native manual fetch) ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
include(FetchContent)

# 1. Fetch wgpu-native binaries (Windows)
FetchContent_Declare(
    wgpu_native
    URL https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-windows-x86_64-msvc-release.zip
)
# Binaries don't have CMakeLists.txt, so we populate only
FetchContent_GetProperties(wgpu_native)
if(NOT wgpu_native_POPULATED)
    FetchContent_Populate(wgpu_native)
endif()

# 2. Fetch C++ Wrapper (headers)
FetchContent_Declare(
    webgpu_cpp
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-Cpp
    GIT_TAG        main
)
FetchContent_MakeAvailable(webgpu_cpp)

# 3. Fetch stb (for image loading)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# 4. Fetch abseil-cpp
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        20230802.1 # LTS 20230802.1
)
FetchContent_MakeAvailable(absl)

# 6. Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# 5. Create 'webgpu' target
add_library(webgpu INTERFACE)

# Includes
# Includes
# Use headers from WebGPU-Cpp repo to ensure compatibility with the wrapper
set(WEB_GPU_CPP_INC "${CMAKE_BINARY_DIR}/webgpu_cpp_include")
file(MAKE_DIRECTORY "${WEB_GPU_CPP_INC}/webgpu")

configure_file(
    "${webgpu_cpp_SOURCE_DIR}/wgpu-native/webgpu.hpp"
    "${WEB_GPU_CPP_INC}/webgpu/webgpu.hpp"
    COPYONLY
)
configure_file(
    "${wgpu_native_SOURCE_DIR}/include/webgpu/webgpu.h"
    "${WEB_GPU_CPP_INC}/webgpu/webgpu.h"
    COPYONLY
)
if(EXISTS "${webgpu_cpp_SOURCE_DIR}/wgpu-native/wgpu.h")
    configure_file(
        "${webgpu_cpp_SOURCE_DIR}/wgpu-native/wgpu.h"
        "${WEB_GPU_CPP_INC}/webgpu/wgpu.h"
        COPYONLY
    )
endif()

target_include_directories(webgpu INTERFACE "${WEB_GPU_CPP_INC}")

# 3. stb (stb_image.h is at root)
target_include_directories(webgpu INTERFACE "${stb_SOURCE_DIR}")

# Libraries
# wgpu-native static import lib for dll
set(WGPU_LIB "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.lib")
if(NOT EXISTS "${WGPU_LIB}")
    # Try .dll.lib
    set(WGPU_LIB "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll.lib")
endif()
target_link_libraries(webgpu INTERFACE "${WGPU_LIB}")

# Defines (Ensure we use WGPU backend if wrapper uses it, though mostly needed for Dawn vs WGPU checks in code)
target_compile_definitions(webgpu INTERFACE WEBGPU_BACKEND_WGPU)

# Helper to copy DLL
function(target_copy_webgpu_binaries target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll"
        "$<TARGET_FILE_DIR:${target}>"
    )
endfunction()

# --- Embedding Shaders ---
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../src/shaders/*.wgsl")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_shaders.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_shaders.py 
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/shaders 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_shaders.h
    DEPENDS ${SHADER_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_shaders.py
    COMMENT "Embedding shaders"
)

# --- Sources ---
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "benchmark_main\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main\\.cpp") 

file(GLOB HEADERS "src/*.h")
list(APPEND HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_shaders.h")

# --- Core Library ---
# Create a static library to share compilation units between exe and python module
add_library(websiftgpu_core STATIC ${SOURCES} ${HEADERS})
target_include_directories(websiftgpu_core PUBLIC src)
target_link_libraries(websiftgpu_core PUBLIC 
    webgpu
    ws2_32 userenv bcrypt opengl32 gdi32 user32 propsys runtimeobject ntdll Threads::Threads
)
# Ensure PIC for shared lib linking
set_property(TARGET websiftgpu_core PROPERTY POSITION_INDEPENDENT_CODE ON)


add_executable(websiftgpu_cpp src/main.cpp)
target_link_libraries(websiftgpu_cpp PRIVATE websiftgpu_core)
target_copy_webgpu_binaries(websiftgpu_cpp)

# --- Python Subdirectory ---
# Pass the WGPU_LIB and other vars if needed, but they should be visible if defined in parent.
# We need to ensure 'webgpu' target is visible. It is (INTERFACE).
# We also need 'websiftgpu_core' to be visible. It is.
add_subdirectory(../python ${CMAKE_CURRENT_BINARY_DIR}/python_build)


# Threads often needed
find_package(Threads REQUIRED)
target_link_libraries(websiftgpu_cpp PRIVATE Threads::Threads)

# Benchmark
# Benchmark
add_executable(websiftgpu_bench src/benchmark_main.cpp)
target_link_libraries(websiftgpu_bench PRIVATE websiftgpu_core absl::time)
target_copy_webgpu_binaries(websiftgpu_bench)


# Tests
# Tests
add_executable(websiftgpu_test tests/test_main.cpp)
target_link_libraries(websiftgpu_test PRIVATE websiftgpu_core)
target_copy_webgpu_binaries(websiftgpu_test)


# --- Invariance Test (OpenCV Required) ---
# --- Invariance & Matcher Tests (OpenCV Required) ---
find_package(OpenCV QUIET)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Building invariance and matcher tests.")
    include_directories(${OpenCV_INCLUDE_DIRS})

    add_executable(websiftgpu_invariance tests/test_invariance.cpp)
    target_link_libraries(websiftgpu_invariance PRIVATE websiftgpu_core ${OpenCV_LIBS})
    target_copy_webgpu_binaries(websiftgpu_invariance)
else()
    message(WARNING "OpenCV NOT found. Skipping build of websiftgpu_invariance. Please install OpenCV to build these tests.")
endif()

# Matcher Test (No OpenCV dependency)
add_executable(websiftgpu_test_matcher tests/test_matcher.cpp)
target_link_libraries(websiftgpu_test_matcher PRIVATE websiftgpu_core)
target_copy_webgpu_binaries(websiftgpu_test_matcher)
