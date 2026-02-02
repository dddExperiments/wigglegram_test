#ifndef WEBSIFTGPU_CPP_SRC_SIFT_BASE_H_
#define WEBSIFTGPU_CPP_SRC_SIFT_BASE_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <webgpu/webgpu.hpp>

struct Keypoint {
    float x;
    float y;
    float octave;
    float scale;
    float sigma;
    float orientation;
    // Padding to match GPU struct alignment if read directly, 
    // but this is the host struct, so we can pack it or keep it simple.
};

struct SIFTOptions {
    bool quantizeDescriptors = false;
    float contrastThreshold = 0.03f;
    float edgeThreshold = 10.0f;
};

class SIFTBase {
 public:
    SIFTBase();
    virtual ~SIFTBase();

    virtual void Init(wgpu::Device device, const SIFTOptions& options = SIFTOptions());
    virtual void Resize(int width, int height) = 0;
    virtual void DetectKeypoints(const uint8_t* image_data, int width, int height) = 0;
    
    const std::vector<Keypoint>& GetKeypoints() const { return keypoints_; }
    size_t GetKeypointsCount() const { return keypoints_.size(); }

 protected:
    wgpu::Device device_;
    wgpu::Queue queue_;
    SIFTOptions options_;

    int width_;
    int height_;
    
    std::vector<Keypoint> keypoints_;
    
    // Helper methods
    wgpu::ShaderModule CreateShaderModule(const std::string& source);
    wgpu::Buffer createBuffer(size_t size, wgpu::BufferUsage usage);

    float GetSigma(int s);
    std::vector<float> CreateKernel(float sigma, int radius);
    wgpu::Buffer GetKernelBuffer(float sigma, int radius);

    std::map<std::string, wgpu::Buffer> kernel_cache_;
};

#endif  // WEBSIFTGPU_CPP_SRC_SIFT_BASE_H_
