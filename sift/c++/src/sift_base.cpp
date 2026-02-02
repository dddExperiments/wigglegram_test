#include "sift_base.h"
#include <iostream>

SIFTBase::SIFTBase() 
    : width_(0), height_(0) {}

SIFTBase::~SIFTBase() {}

void SIFTBase::Init(wgpu::Device d, const SIFTOptions& options) {
    device_ = d;
    queue_ = d.getQueue();
    options_ = options;
}

wgpu::ShaderModule SIFTBase::CreateShaderModule(const std::string& source) {
    wgpu::ShaderSourceWGSL wgsl_desc = {};
    wgsl_desc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgsl_desc.code = wgpu::StringView(source.c_str());

    wgpu::ShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&wgsl_desc);

    return device_.createShaderModule(descriptor);
}

wgpu::Buffer SIFTBase::createBuffer(size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor descriptor;
    descriptor.size = (size + 3) & ~3; // Align to 4 bytes
    descriptor.usage = usage;
    descriptor.mappedAtCreation = false;
    return device_.createBuffer(descriptor);
}

float SIFTBase::GetSigma(int s) {
    return 1.6f * std::pow(2.0f, (float)s / 3.0f); // Default values
}

std::vector<float> SIFTBase::CreateKernel(float sigma, int radius) {
    int size = radius * 2 + 1;
    std::vector<float> kernel(size);
    float sum = 0;
    for (int i = -radius; i <= radius; i++) {
        float v = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    for (int i = 0; i < size; i++) kernel[i] /= sum;
    return kernel;
}

wgpu::Buffer SIFTBase::GetKernelBuffer(float sigma, int radius) {
    char key[64];
    snprintf(key, sizeof(key), "%.4f_%d", sigma, radius);
    std::string skey(key);

    if (kernel_cache_.count(skey)) return kernel_cache_[skey];

    std::vector<float> kernel = CreateKernel(sigma, radius);
    wgpu::Buffer buffer = createBuffer(kernel.size() * sizeof(float), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    queue_.writeBuffer(buffer, 0, kernel.data(), kernel.size() * sizeof(float));

    kernel_cache_[skey] = buffer;
    return buffer;
}
