#ifndef WEBSIFTGPU_CPP_SRC_SIFT_MATCHER_H_
#define WEBSIFTGPU_CPP_SRC_SIFT_MATCHER_H_

#include <vector>
#include <webgpu/webgpu.hpp>

struct Match {
    int trainIdx; // Index in descriptorsB
    int queryIdx; // Index in descriptorsA
    float distance;
};

class SIFTMatcher {
 public:
    SIFTMatcher();
    ~SIFTMatcher();

    void Init(wgpu::Device device);
    
    // Returns indices in descB that match descA
    // ratio_threshold: Lowe's ratio test (e.g. 0.75)
    std::vector<Match> MatchDescriptors(const std::vector<float>& descA, 
                                        const std::vector<float>& descB, 
                                        float ratio_threshold = 0.75f,
                                        bool quantize = false);
    
    // Guided matching with F-matrix
    // keypoints are flattened [x0, y0, x1, y1...]
    // F is 3x3 array (row-major or col-major? standard is row-major usually but we pass as raw floats)
    // threshold in pixels
    std::vector<Match> MatchGuided(const std::vector<float>& descA, const std::vector<float>& kpsA,
                                   const std::vector<float>& descB, const std::vector<float>& kpsB,
                                   const std::vector<float>& F,
                                   float threshold,
                                   float ratio_threshold = 0.75f);

 private:
    wgpu::Device device_;
    wgpu::Queue queue_;
    wgpu::ComputePipeline pipeline_;
    wgpu::ComputePipeline pipeline_quant_;
    wgpu::ComputePipeline pipeline_guided_; // New pipeline
    
    wgpu::Buffer params_buf_;
    
    // Helpers
    wgpu::Buffer createBuffer(size_t size, wgpu::BufferUsage usage);
    std::string loadShader(const std::string& name); // Updated to take name
};

#endif // WEBSIFTGPU_CPP_SRC_SIFT_MATCHER_H_
