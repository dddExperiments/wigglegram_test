#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace utils {

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<uint8_t> loadImage(const std::string& path, int* width, int* height) {
    int channels;
    // Force 4 channels (RGBA)
    unsigned char* data = stbi_load(path.c_str(), width, height, &channels, 4); 
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    std::vector<uint8_t> result(data, data + (*width) * (*height) * 4);
    stbi_image_free(data);
    return result;
}

void saveImage(const std::string& path, const uint8_t* data, int width, int height) {
    stbi_write_png(path.c_str(), width, height, 4, data, width * 4);
}

}  // namespace utils
