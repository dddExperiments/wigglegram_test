#ifndef WEBSIFTGPU_CPP_SRC_SIFT_PACKED_H_
#define WEBSIFTGPU_CPP_SRC_SIFT_PACKED_H_

#include "sift_base.h"

#include <chrono>

struct SIFTProfiling {
    double total_ms = 0;
    double upload_ms = 0;
    double grayscale_ms = 0;
    double pyramids_ms = 0;
    double extrema_ms = 0;
    double orientation_ms = 0;
    double descriptor_ms = 0;
    double download_ms = 0;
};

class SIFTPacked : public SIFTBase {
 public:
    SIFTPacked();
    ~SIFTPacked(); // override removed as per instruction

    void Init(wgpu::Device device, const SIFTOptions& options = SIFTOptions()) override;
    void Resize(int width, int height) override;
    void DetectKeypoints(const uint8_t* image_data, int width, int height) override;
    void ReadbackDescriptors(std::vector<float>& out_descriptors); // Moved to public

    const SIFTProfiling& GetProfiling() const { return profiling_; }

 private:
    SIFTProfiling profiling_;
    void InitPipelines();
    void InitBuffers();
    
    std::string loadShader(const std::string& filename);
    
    // Dispatchers
    void RunGrayscale(wgpu::Texture output_tex, int pw, int ph);
    void BuildPyramids(int pw, int ph);
    void RunBlur(wgpu::Texture in_tex, wgpu::Texture out_tex, wgpu::Texture temp_tex, int w, int h, float sigma);
    void RunDownsample(wgpu::Texture in_tex, wgpu::Texture out_tex, int sw, int sh, int dw, int dh);
    void RunDoG(wgpu::Texture tex_a, wgpu::Texture tex_b, wgpu::Texture out_tex, int w, int h);
    void DetectExtrema();
    void ComputeOrientations();
    void PrepareDispatch(); // Prepares indirect dispatch buffer
    void RunComputeDescriptors();
    void ReadbackKeypoints();

    void WriteTimestamp(uint32_t index);


    // Data
    struct PyramidCache {
        int w, h;
        wgpu::Texture base_texture;
        wgpu::Texture temp_texture;
        std::vector<std::vector<wgpu::Texture>> gaussian_pyramid;
        std::vector<std::vector<wgpu::Texture>> dog_pyramid;
        std::vector<std::pair<int, int>> octave_sizes;
    };
    std::unique_ptr<PyramidCache> pyramid_cache_;

    // Pipelines
    wgpu::ComputePipeline pipeline_grayscale_;
    wgpu::ComputePipeline pipeline_blur_h_;
    wgpu::ComputePipeline pipeline_blur_v_;
    wgpu::ComputePipeline pipeline_dog_;
    wgpu::ComputePipeline pipeline_downsample_;
    wgpu::ComputePipeline pipeline_extrema_;
    wgpu::ComputePipeline pipeline_orientation_;
    wgpu::ComputePipeline pipeline_descriptor_;
    wgpu::ComputePipeline pipeline_prepare_dispatch_;

    struct Buffers {
        wgpu::Buffer keypoints;
        wgpu::Buffer descriptors;
        wgpu::Buffer params16;
        wgpu::Buffer params_extrema;
        wgpu::Buffer indirect_dispatch; // For dispatchWorkgroupsIndirect
    } buffers_;

    wgpu::Texture input_texture_;
    
    // Timestamp Query
    wgpu::QuerySet query_set_;
    wgpu::Buffer query_resolve_buf_;
    wgpu::Buffer query_result_buf_;
    float timestamp_period_ = 1.0f;

    // Constants
    static constexpr int kNumOctaves = 4;
    static constexpr int kScalesPerOctave = 3;
    static constexpr float kSigmaBase = 1.6f;
    static constexpr float kContrastThreshold = 0.03f;
    static constexpr float kEdgeThreshold = 10.0f;
    static constexpr int kMaxKeypoints = 100000;
};

#endif  // WEBSIFTGPU_CPP_SRC_SIFT_PACKED_H_
