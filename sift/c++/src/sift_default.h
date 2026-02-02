#ifndef WEBSIFTGPU_CPP_SRC_SIFT_DEFAULT_H_
#define WEBSIFTGPU_CPP_SRC_SIFT_DEFAULT_H_

#include "sift_base.h"

class SIFTDefault : public SIFTBase {
 public:
    SIFTDefault();
    ~SIFTDefault() override;

    void Init(wgpu::Device device, const SIFTOptions& options = SIFTOptions()) override;
    void Resize(int width, int height) override;
    void DetectKeypoints(const uint8_t* image_data, int width, int height) override;

 private:
    void InitPipelines();
    void InitBuffers();
    
    // Shader Loaders
    std::string loadShader(const std::string& filename);

    
    // Dispatchers
    void RunGrayscale(wgpu::Texture output_tex);
    void BuildPyramids();
    void RunBlur(wgpu::Texture in_tex, wgpu::Texture out_tex, wgpu::Texture temp_tex, int w, int h, float sigma);
    void RunDownsample(wgpu::Texture in_tex, wgpu::Texture out_tex, int sw, int sh, int dw, int dh);
    void RunDoG(wgpu::Texture tex_a, wgpu::Texture tex_b, wgpu::Texture out_tex, int w, int h);
    void DetectExtrema();
    void RunPrepareDispatch();
    void ComputeOrientations();
    void ReadbackKeypoints();

    // Data
    struct PyramidCache {
        int w, h;
        wgpu::Texture base_texture;
        wgpu::Texture temp_texture;
        std::vector<std::vector<wgpu::Texture>> gaussian_pyramid;
        std::vector<std::vector<wgpu::Texture>> dog_pyramid;
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

    // Buffers
    struct Buffers {
        wgpu::Buffer keypoints;
        wgpu::Buffer descriptors;
        wgpu::Buffer params16;
        wgpu::Buffer params_extrema;
        wgpu::Buffer indirect_dispatch;
        wgpu::Buffer debug_hist;
    } buffers_;

    wgpu::Texture input_texture_;

    // Constants
    static constexpr int kNumOctaves = 4;
    static constexpr int kScalesPerOctave = 3;
    static constexpr float kSigmaBase = 1.6f;
    static constexpr float kContrastThreshold = 0.03f;
    static constexpr float kEdgeThreshold = 10.0f;
    static constexpr int kMaxKeypoints = 100000;
};

#endif  // WEBSIFTGPU_CPP_SRC_SIFT_DEFAULT_H_
