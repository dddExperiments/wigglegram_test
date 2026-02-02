#ifndef WEBSIFTGPU_CPP_SRC_UTILS_H_
#define WEBSIFTGPU_CPP_SRC_UTILS_H_

#include <string>
#include <vector>
#include <cstdint>

namespace utils {

// Reads entire file into string
std::string readFile(const std::string& path);

// Loads image as RGBA8
std::vector<uint8_t> loadImage(const std::string& path, int* width, int* height);

// Saves RGBA8 image to PNG
void saveImage(const std::string& path, const uint8_t* data, int width, int height);

}  // namespace utils

#endif  // WEBSIFTGPU_CPP_SRC_UTILS_H_
