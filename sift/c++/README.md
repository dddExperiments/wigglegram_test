# WebSiftGPU C++ Implementation

This directory contains a C++ port of WebSiftGPU using `wgpu-native` and the `WebGPU-Cpp` wrapper.

## Prerequisites
- CMake 3.16+
- C++17 Compiler
- Windows (Currently setup for prebuilt `wgpu-native` binaries on Windows)
- OpenCV (Required for Invariance and Matcher tests)
  - Download Windows binaries from [opencv.org/releases](https://opencv.org/releases/) (e.g., 4.x.x)
  - Set `OpenCV_DIR` environment variable to `path/to/build` (containing `OpenCVConfig.cmake`).
  - Or run cmake with `-DOpenCV_DIR="C:/path/to/opencv/build"`.

## Structure
- `src/`: Source code
  - `main.cpp`: CLI entry point.
  - `sift_base.h/cpp`: Base class for SIFT implementations.
  - `sift_default.h/cpp`: Standard unpacked SIFT implementation.
  - `sift_packed.h/cpp`: Optimized packed SIFT implementation.
  - `utils.h/cpp`: Image loading/saving helpers (stb_image).
  - `benchmark_main.cpp`: Performance benchmarking tool.
- `tests/`: Unit tests.
  - `test_main.cpp`: Simple verification tests.
- `CMakeLists.txt`: CMake build configuration.

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This will download `wgpu-native` binaries and `WebGPU-Cpp` headers automatically.

## Usage

### 1. Main CLI Tool
Run the SIFT detector on an image (defaults to packed implementation):
```bash
./bin/Release/websiftgpu_cpp.exe <path/to/image.jpg>
```
Ensure `wgpu_native.dll` and the `shaders/` folder are in the working directory (or `shaders` is at `../../shaders`).

### 2. Benchmark
Run the profiling benchmark to measure runtime of each pipeline stage:
```bash
./bin/Release/websiftgpu_bench.exe <path/to/image.jpg>
```

### 3. Tests
Run the unit test suite:
```bash
./bin/Release/websiftgpu_test.exe
```

## Google Style
The code follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
