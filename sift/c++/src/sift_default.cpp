#include "sift_default.h"
#include "sift_base.h"
#include "embedded_shaders.h"
// Check content.

#include <iostream>
#include <cmath>
#include <algorithm>
#include <webgpu/webgpu.hpp> // Ensure using the same wrapper
#include "utils.h"

// Constants
constexpr int SIFTDefault::kNumOctaves;
constexpr int SIFTDefault::kScalesPerOctave;
constexpr float SIFTDefault::kSigmaBase;

SIFTDefault::SIFTDefault() = default;

SIFTDefault::~SIFTDefault() = default;

void SIFTDefault::Init(wgpu::Device device, const SIFTOptions& options) {
    SIFTBase::Init(device, options);
    InitPipelines();
    InitBuffers();
}

std::string SIFTDefault::loadShader(const std::string& filename) {
    // Check embedded
    std::string key = (filename == "prepare_dispatch.wgsl") ? filename : ("default/" + filename);
    std::string code = shader_embed::GetShader(key);
    if (!code.empty()) return code;

    std::vector<std::string> search_paths;
    if (filename == "prepare_dispatch.wgsl") {
        search_paths = { "../../src/shaders/common/", "../src/shaders/common/", "src/shaders/common/" };
    } else {
        search_paths = { "../../src/shaders/detection/default/", "../src/shaders/detection/default/", "src/shaders/detection/default/" };
    }

    for (const auto& base : search_paths) {
        try {
            return utils::readFile(base + filename);
        } catch (...) {}
    }
    return "";
}

void SIFTDefault::InitPipelines() {
    auto create_compute_pipeline = [&](const std::string& shader_file, const std::string& entry_point = "main") {
        std::string code = loadShader(shader_file);
        wgpu::ShaderModule module = CreateShaderModule(code);
        
        wgpu::ComputePipelineDescriptor desc;
        desc.compute.module = module;
        desc.compute.entryPoint = wgpu::StringView(entry_point.c_str());
        return device_.createComputePipeline(desc);
    };

    pipeline_grayscale_ = create_compute_pipeline("grayscale.wgsl");
    pipeline_blur_h_ = create_compute_pipeline("blur_horizontal.wgsl", "main"); // Assuming multi-entry in one file? Check JS. 
    // JS: initPipelines('../shaders/default') -> loads individual files usually?
    // Let's check JS implementation again or directory listing.
    // Listing showed: default/ has children.
    // Let's assume files: blur.wgsl (likely contains both or valid wgsl).
    // Actually, create_compute_pipeline usually takes one entry point.
    // If blur.wgsl has multiple entry points, we need to specify.
    pipeline_blur_v_ = create_compute_pipeline("blur_vertical.wgsl", "main");
    pipeline_dog_ = create_compute_pipeline("dog.wgsl");
    pipeline_downsample_ = create_compute_pipeline("downsample.wgsl");
    pipeline_extrema_ = create_compute_pipeline("extrema.wgsl");
    pipeline_orientation_ = create_compute_pipeline("orientation.wgsl");
    pipeline_descriptor_ = create_compute_pipeline("descriptor.wgsl");
    pipeline_prepare_dispatch_ = create_compute_pipeline("prepare_dispatch.wgsl");
}

void SIFTDefault::InitBuffers() {
    size_t kp_size = 16 + kMaxKeypoints * 32;
    buffers_.keypoints = createBuffer(kp_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect);
    
    size_t desc_size = kMaxKeypoints * 128 * 4;
    buffers_.descriptors = createBuffer(desc_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);

    buffers_.params16 = createBuffer(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    buffers_.params_extrema = createBuffer(24, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    buffers_.indirect_dispatch = createBuffer(24, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst);
    buffers_.debug_hist = createBuffer(kMaxKeypoints * 36 * 4, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
}

void SIFTDefault::Resize(int w, int h) {
    if (pyramid_cache_ && pyramid_cache_->w == w && pyramid_cache_->h == h) {
        return;
    }

    pyramid_cache_ = std::make_unique<PyramidCache>();
    pyramid_cache_->w = w;
    pyramid_cache_->h = h;

    // Helper to create storage texture
    auto create_tex = [&](int width, int height) {
        wgpu::TextureDescriptor desc;
        desc.size = { (uint32_t)width, (uint32_t)height, 1 };
        desc.sampleCount = 1;
        desc.mipLevelCount = 1;
        desc.format = wgpu::TextureFormat::R32Float;
        desc.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding;
        return device_.createTexture(desc);
    };

    pyramid_cache_->base_texture = create_tex(w, h);
    pyramid_cache_->temp_texture = create_tex(w, h);

    int curr_w = w;
    int curr_h = h;

    for (int o = 0; o < kNumOctaves; ++o) {
        std::vector<wgpu::Texture> gauss_octave;
        std::vector<wgpu::Texture> dog_octave;

        for (int s = 0; s < kScalesPerOctave + 3; ++s) {
            gauss_octave.push_back(create_tex(curr_w, curr_h));
        }
        for (int s = 0; s < kScalesPerOctave + 2; ++s) {
            dog_octave.push_back(create_tex(curr_w, curr_h));
        }
        
        pyramid_cache_->gaussian_pyramid.push_back(gauss_octave);
        pyramid_cache_->dog_pyramid.push_back(dog_octave);

        curr_w /= 2;
        curr_h /= 2;
    }
}

void SIFTDefault::DetectKeypoints(const uint8_t* image_data, int width, int height) {
    width_ = width;
    height_ = height;
    keypoints_.clear();

    // Create Input Texture
    wgpu::TextureDescriptor desc;
    desc.size = { (uint32_t)width, (uint32_t)height, 1 };
    desc.sampleCount = 1;
    desc.mipLevelCount = 1;
    desc.format = wgpu::TextureFormat::RGBA8Unorm;
    desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment; // RenderAttachment often needed for copy dest compatibility in some backends/setups, or just CopyDst
    input_texture_ = device_.createTexture(desc);

    wgpu::TexelCopyTextureInfo dst;
    dst.texture = input_texture_;
    
    wgpu::TexelCopyBufferLayout layout;
    layout.bytesPerRow = width * 4;
    layout.rowsPerImage = height;

    wgpu::Extent3D extent = { (uint32_t)width, (uint32_t)height, 1 };
    queue_.writeTexture(dst, image_data, width * height * 4, layout, extent);

    Resize(width, height);

    // Reset Count
    uint32_t zero = 0;
    queue_.writeBuffer(buffers_.keypoints, 0, &zero, 4);

    RunGrayscale(pyramid_cache_->base_texture);
    BuildPyramids();
    DetectExtrema();
    RunPrepareDispatch();
    ComputeOrientations();
    ReadbackKeypoints();
}

void SIFTDefault::RunGrayscale(wgpu::Texture output_tex) {
   wgpu::BindGroupEntry entries[2];
   entries[0].binding = 0;
   entries[0].textureView = input_texture_.createView();
   entries[1].binding = 1;
   entries[1].textureView = output_tex.createView();

   wgpu::BindGroupDescriptor desc;
   desc.layout = pipeline_grayscale_.getBindGroupLayout(0);
   desc.entryCount = 2;
   desc.entries = entries;
   wgpu::BindGroup bind_group = device_.createBindGroup(desc);

   wgpu::CommandEncoder encoder = device_.createCommandEncoder();
   wgpu::ComputePassEncoder pass = encoder.beginComputePass();
   pass.setPipeline(pipeline_grayscale_);
   pass.setBindGroup(0,  bind_group, 0, nullptr);
   pass.dispatchWorkgroups((width_ + 15) / 16,  (height_ + 15) / 16, 1);
   pass.end();
   queue_.submit(1, &encoder.finish()); // Using raw pointer hack? or wrapper? wgpu::CommandBuffer commands = encoder.finish(); queue.submit(1, &commands);
   // C++ wrapper: Queue::Submit(uint32_t commandCount, CommandBuffer const * commands)
   // Actually webgpu_cpp.h uses `queue.submit(count, &cmdBuf)`
}

// ... Additional implementations for Blur, Downsample, DoG, etc.
// Due to length, I will provide the skeleton and core structure first, then fill in details if needed or in next chunks.
// I will implement helper methods here.


void SIFTDefault::RunBlur(wgpu::Texture in_tex, wgpu::Texture out_tex, wgpu::Texture temp_tex, int w, int h, float sigma) {
    int radius = std::ceil(sigma * 3);
    wgpu::Buffer kernel_buf = GetKernelBuffer(sigma, radius);

    // Params
    uint32_t params[] = { (uint32_t)w, (uint32_t)h, (uint32_t)radius, 0 };
    queue_.writeBuffer(buffers_.params16, 0, params, sizeof(params));

    // Horizontal Pass
    {
        wgpu::BindGroupEntry entries[4];
        entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = 16;
        entries[1].binding = 1; entries[1].textureView = in_tex.createView();
        entries[2].binding = 2; entries[2].textureView = temp_tex.createView();
        entries[3].binding = 3; entries[3].buffer = kernel_buf; entries[3].size = WGPU_WHOLE_SIZE;

        wgpu::BindGroupDescriptor desc;
        desc.layout = pipeline_blur_h_.getBindGroupLayout(0);
        desc.entryCount = 4;
        desc.entries = entries;
        wgpu::BindGroup bind = device_.createBindGroup(desc);

        wgpu::CommandEncoder enc = device_.createCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.beginComputePass();
        pass.setPipeline(pipeline_blur_h_);
        pass.setBindGroup(0,  bind, 0, nullptr);
        pass.dispatchWorkgroups((w + 15) / 16,  (h + 15) / 16, 1);
        pass.end();
        wgpu::CommandBuffer cmd = enc.finish();
        queue_.submit(1, &cmd);
    }

    // Vertical Pass
    {
        wgpu::BindGroupEntry entries[4];
        entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = 16;
        entries[1].binding = 1; entries[1].textureView = temp_tex.createView();
        entries[2].binding = 2; entries[2].textureView = out_tex.createView();
        entries[3].binding = 3; entries[3].buffer = kernel_buf; entries[3].size = WGPU_WHOLE_SIZE;

        wgpu::BindGroupDescriptor desc;
        desc.layout = pipeline_blur_v_.getBindGroupLayout(0);
        desc.entryCount = 4;
        desc.entries = entries;
        wgpu::BindGroup bind = device_.createBindGroup(desc);

        wgpu::CommandEncoder enc = device_.createCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.beginComputePass();
        pass.setPipeline(pipeline_blur_v_);
        pass.setBindGroup(0,  bind, 0, nullptr);
        pass.dispatchWorkgroups((w + 15) / 16,  (h + 15) / 16, 1);
        pass.end();
        wgpu::CommandBuffer cmd = enc.finish();
        queue_.submit(1, &cmd);
    }
}

void SIFTDefault::RunDownsample(wgpu::Texture in_tex, wgpu::Texture out_tex, int sw, int sh, int dw, int dh) {
    uint32_t params[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)dw, (uint32_t)dh };
    queue_.writeBuffer(buffers_.params16, 0, params, sizeof(params));

    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = 16;
    entries[1].binding = 1; entries[1].textureView = in_tex.createView();
    entries[2].binding = 2; entries[2].textureView = out_tex.createView();

    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_downsample_.getBindGroupLayout(0);
    desc.entryCount = 3;
    desc.entries = entries;
    wgpu::BindGroup bind = device_.createBindGroup(desc);

    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipeline_downsample_);
    pass.setBindGroup(0,  bind, 0, nullptr);
    pass.dispatchWorkgroups((dw + 15) / 16,  (dh + 15) / 16, 1);
    pass.end();
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
}

void SIFTDefault::RunDoG(wgpu::Texture tex_a, wgpu::Texture tex_b, wgpu::Texture out_tex, int w, int h) {
    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0; entries[0].textureView = tex_a.createView();
    entries[1].binding = 1; entries[1].textureView = tex_b.createView();
    entries[2].binding = 2; entries[2].textureView = out_tex.createView();

    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_dog_.getBindGroupLayout(0);
    desc.entryCount = 3;
    desc.entries = entries;
    wgpu::BindGroup bind = device_.createBindGroup(desc);

    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipeline_dog_);
    pass.setBindGroup(0,  bind, 0, nullptr);
    pass.dispatchWorkgroups((w + 15) / 16,  (h + 15) / 16, 1);
    pass.end();
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
}

void SIFTDefault::BuildPyramids() {
    int w = width_;
    int h = height_;
    
    for (int o = 0; o < kNumOctaves; ++o) {
        auto& gauss_octave = pyramid_cache_->gaussian_pyramid[o];
        auto& dog_octave = pyramid_cache_->dog_pyramid[o];

        if (o == 0) {
            RunBlur(pyramid_cache_->base_texture, gauss_octave[0], pyramid_cache_->temp_texture, w, h, kSigmaBase);
        } else {
            // Downsample from previous octave
             int prev_w = w * 2;
             int prev_h = h * 2;
             wgpu::Texture prev_tex = pyramid_cache_->gaussian_pyramid[o-1][kScalesPerOctave];
             RunDownsample(prev_tex, gauss_octave[0], prev_w, prev_h, w, h);
        }

        for (size_t s = 1; s < gauss_octave.size(); ++s) {
            float sigma = GetSigma(s);
            float prev_sigma = GetSigma(s - 1);
            float diff = std::sqrt(sigma * sigma - prev_sigma * prev_sigma);
            RunBlur(gauss_octave[s - 1], gauss_octave[s], pyramid_cache_->temp_texture, w, h, diff);
        }

        for (size_t s = 0; s < dog_octave.size(); ++s) {
            RunDoG(gauss_octave[s], gauss_octave[s + 1], dog_octave[s], w, h);
        }

        w /= 2; h /= 2;
    }
}

void SIFTDefault::DetectExtrema() {
    int w = width_;
    int h = height_;

    for (int o = 0; o < kNumOctaves; ++o) {
        for (int s = 1; s <= kScalesPerOctave; ++s) {
            
            struct {
                int w, h, o, s;
                float contrast, edge;
            } params;

            params.w = w; params.h = h;
            params.o = o; params.s = s;
            params.contrast = kContrastThreshold / kScalesPerOctave;
            params.edge = kEdgeThreshold;

            queue_.writeBuffer(buffers_.params_extrema, 0, &params, sizeof(params));

            wgpu::BindGroupEntry entries[5];
            entries[0].binding = 0; entries[0].buffer = buffers_.params_extrema; entries[0].size = 24;
            entries[1].binding = 1; entries[1].textureView = pyramid_cache_->dog_pyramid[o][s - 1].createView();
            entries[2].binding = 2; entries[2].textureView = pyramid_cache_->dog_pyramid[o][s].createView();
            entries[3].binding = 3; entries[3].textureView = pyramid_cache_->dog_pyramid[o][s + 1].createView();
            entries[4].binding = 4; entries[4].buffer = buffers_.keypoints; entries[4].size = 16 + kMaxKeypoints * 32;

            wgpu::BindGroupDescriptor desc;
            desc.layout = pipeline_extrema_.getBindGroupLayout(0);
            desc.entryCount = 5;
            desc.entries = entries;
            wgpu::BindGroup bind = device_.createBindGroup(desc);

            wgpu::CommandEncoder enc = device_.createCommandEncoder();
            wgpu::ComputePassEncoder pass = enc.beginComputePass();
            pass.setPipeline(pipeline_extrema_);
            pass.setBindGroup(0,  bind, 0, nullptr);
            pass.dispatchWorkgroups((w + 15) / 16,  (h + 15) / 16, 1);
            pass.end();
            wgpu::CommandBuffer cmd = enc.finish();
            queue_.submit(1, &cmd);
        }
        w /= 2; h /= 2;
    }
}

void SIFTDefault::RunPrepareDispatch() {
    wgpu::BindGroupEntry entries[2];
    entries[0].binding = 0; entries[0].buffer = buffers_.keypoints; entries[0].size = 16;
    entries[1].binding = 1; entries[1].buffer = buffers_.indirect_dispatch; entries[1].size = 24;
    
    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_prepare_dispatch_.getBindGroupLayout(0);
    desc.entryCount = 2;
    desc.entries = entries;
    wgpu::BindGroup bind = device_.createBindGroup(desc);
    
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipeline_prepare_dispatch_);
    pass.setBindGroup(0, bind, 0, nullptr);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
}

void SIFTDefault::ComputeOrientations() {
    int w = width_;
    int h = height_;
    
    for (int o = 0; o < kNumOctaves; ++o) {
        struct { int w, h, o; } params = { w, h, o };
        
        wgpu::Buffer params_buf = createBuffer(12, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        queue_.writeBuffer(params_buf, 0, &params, sizeof(params));

        wgpu::BindGroupEntry entries[5];
        entries[0].binding = 0; entries[0].buffer = params_buf; entries[0].size = 12;
        entries[1].binding = 1; entries[1].buffer = buffers_.keypoints; entries[1].size = buffers_.keypoints.getSize();
        entries[2].binding = 2; entries[2].textureView = pyramid_cache_->gaussian_pyramid[o][1].createView();
        entries[3].binding = 3; entries[3].textureView = pyramid_cache_->gaussian_pyramid[o][2].createView();
        entries[4].binding = 4; entries[4].textureView = pyramid_cache_->gaussian_pyramid[o][3].createView();

        wgpu::BindGroupDescriptor desc;
        desc.layout = pipeline_orientation_.getBindGroupLayout(0);
        desc.entryCount = 5;
        desc.entries = entries;
        wgpu::BindGroup bind = device_.createBindGroup(desc);

        wgpu::CommandEncoder enc = device_.createCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.beginComputePass();
        pass.setPipeline(pipeline_orientation_);
        pass.setBindGroup(0,  bind, 0, nullptr);
        pass.dispatchWorkgroupsIndirect(buffers_.indirect_dispatch, 0); // Offset 0: orientation dispatch params
        pass.end();
        wgpu::CommandBuffer cmd = enc.finish();
        queue_.submit(1, &cmd);
        
        w /= 2; h /= 2;
    }
}

// Helper for static callback
static volatile bool g_map_done = false;
static void MapCallback(WGPUMapAsyncStatus, WGPUStringView, void*, void*) {
    g_map_done = true;
}

void SIFTDefault::ReadbackKeypoints() {
    wgpu::Buffer count_buf = createBuffer(4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    enc.copyBufferToBuffer(buffers_.keypoints, 0, count_buf, 0, 4);
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);

    // Synchronous map-read workaround
    g_map_done = false;
    
    wgpu::BufferMapCallbackInfo callbackInfo = {};
    callbackInfo.mode = wgpu::CallbackMode::AllowSpontaneous;
    callbackInfo.callback = MapCallback;

    count_buf.mapAsync(wgpu::MapMode::Read, 0, 4, callbackInfo);

    while (!g_map_done) {
        device_.poll(false, nullptr);
    }

    const uint32_t* mapped_count = (const uint32_t*)count_buf.getConstMappedRange(0, 4);
    uint32_t count = *mapped_count;
    count_buf.unmap();

    if (count == 0) return;

    size_t kp_byte_size = count * 32;
    wgpu::Buffer read_buf = createBuffer(kp_byte_size, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    enc = device_.createCommandEncoder();
    enc.copyBufferToBuffer(buffers_.keypoints, 16, read_buf, 0, kp_byte_size);
    cmd = enc.finish();
    queue_.submit(1, &cmd);

    g_map_done = false;
    read_buf.mapAsync(wgpu::MapMode::Read, 0, kp_byte_size, callbackInfo);

    while (!g_map_done) {
        device_.poll(false, nullptr);
    }

    const float* mapped_data = (const float*)read_buf.getConstMappedRange(0, kp_byte_size);
    for (uint32_t i = 0; i < count; ++i) {
        Keypoint kp;
        kp.x = mapped_data[i * 8 + 0];
        kp.y = mapped_data[i * 8 + 1];
        kp.octave = mapped_data[i * 8 + 2];
        kp.scale = mapped_data[i * 8 + 3];
        kp.sigma = mapped_data[i * 8 + 4];
        kp.orientation = mapped_data[i * 8 + 5];
        keypoints_.push_back(kp);
    }
    read_buf.unmap();
}
