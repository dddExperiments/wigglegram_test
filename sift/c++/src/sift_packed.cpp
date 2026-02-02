#include "sift_packed.h"
#include "embedded_shaders.h"
#include "utils.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.h>

// wgpu-native extension function
extern "C" float wgpuQueueGetTimestampPeriod(WGPUQueue queue);

constexpr int SIFTPacked::kNumOctaves;
constexpr int SIFTPacked::kScalesPerOctave;
constexpr float SIFTPacked::kSigmaBase;

SIFTPacked::SIFTPacked() = default;
SIFTPacked::~SIFTPacked() = default;

void SIFTPacked::Init(wgpu::Device device, const SIFTOptions& options) {
    SIFTBase::Init(device, options);
    InitPipelines();
    InitBuffers();

    // Get timestamp period (wgpu-native extension)
    // The cast to WGPUQueue is safe for the elie-michel wrapper
    timestamp_period_ = wgpuQueueGetTimestampPeriod((WGPUQueue)queue_);
    if (timestamp_period_ <= 0) timestamp_period_ = 1.0f;
}

std::string SIFTPacked::loadShader(const std::string& filename) {
    // Check if we need quantized descriptor
    std::string final_filename = filename;
    if (filename == "descriptor.wgsl" && options_.quantizeDescriptors) {
        final_filename = "descriptor_quantized.wgsl";
    }

    // Check embedded
    std::string key = (final_filename == "prepare_dispatch.wgsl") ? final_filename : ("packed/" + final_filename);
    std::string code = shader_embed::GetShader(key);
    if (!code.empty()) return code;

    std::vector<std::string> search_paths;
    if (final_filename == "prepare_dispatch.wgsl") {
        search_paths = { "../../src/shaders/common/", "../src/shaders/common/", "src/shaders/common/" };
    } else {
        search_paths = { "../../src/shaders/detection/packed/", "../src/shaders/detection/packed/", "src/shaders/detection/packed/" };
    }

    for (const auto& base : search_paths) {
        try {
            return utils::readFile(base + final_filename);
        } catch (...) {}
    }
    return "";
}

void SIFTPacked::InitPipelines() {
    auto create_compute_pipeline = [&](const std::string& shader_file, const std::string& entry_point = "main") {
        std::string code = loadShader(shader_file);
        wgpu::ShaderModule module = CreateShaderModule(code);
        wgpu::ComputePipelineDescriptor desc;
        desc.compute.module = module;
        desc.compute.entryPoint = wgpu::StringView(entry_point.c_str());
        return device_.createComputePipeline(desc);
    };

    pipeline_grayscale_ = create_compute_pipeline("grayscale.wgsl");
    pipeline_blur_h_ = create_compute_pipeline("blur_horizontal.wgsl", "main");
    pipeline_blur_v_ = create_compute_pipeline("blur_vertical.wgsl", "main");
    pipeline_dog_ = create_compute_pipeline("dog.wgsl");
    pipeline_downsample_ = create_compute_pipeline("downsample.wgsl");
    pipeline_extrema_ = create_compute_pipeline("extrema.wgsl");
    pipeline_orientation_ = create_compute_pipeline("orientation.wgsl");
    pipeline_descriptor_ = create_compute_pipeline("descriptor.wgsl");
    pipeline_prepare_dispatch_ = create_compute_pipeline("prepare_dispatch.wgsl");
}

void SIFTPacked::InitBuffers() {
    size_t kp_size = 16 + kMaxKeypoints * 32;
    buffers_.keypoints = createBuffer(kp_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect);
    
    size_t desc_elem_size = options_.quantizeDescriptors ? 32 * 4 : 128 * 4;
    size_t desc_size = kMaxKeypoints * desc_elem_size;
    buffers_.descriptors = createBuffer(desc_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);

    buffers_.params16 = createBuffer(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    buffers_.params_extrema = createBuffer(24, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    
    // Indirect dispatch buffer: 6 u32 values (orientation x,y,z + descriptor x,y,z)
    buffers_.indirect_dispatch = createBuffer(24, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst);

    // Timestamp Query
    wgpu::QuerySetDescriptor qDesc;
    qDesc.type = wgpu::QueryType::Timestamp;
    qDesc.count = 7; // start, gray, pyr, ext, ori, desc, end
    query_set_ = device_.createQuerySet(qDesc);
    query_resolve_buf_ = createBuffer(7 * 8, wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc);
    query_result_buf_ = createBuffer(7 * 8, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
}

void SIFTPacked::Resize(int w, int h) {
    if (pyramid_cache_ && pyramid_cache_->w == w && pyramid_cache_->h == h) return;

    pyramid_cache_ = std::make_unique<PyramidCache>();
    pyramid_cache_->w = w;
    pyramid_cache_->h = h;

    // Packed Dimensions
    int pw = (w + 1) / 2;
    int ph = (h + 1) / 2;

    auto create_tex = [&](int width, int height) {
        wgpu::TextureDescriptor desc;
        desc.size = { (uint32_t)width, (uint32_t)height, 1 };
        desc.sampleCount = 1;
        desc.mipLevelCount = 1;
        desc.format = wgpu::TextureFormat::RGBA32Float; // Packed uses RGBA32Float for 4 pixels
        desc.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst; // Add CopyDst just in case
        return device_.createTexture(desc);
    };

    pyramid_cache_->base_texture = create_tex(pw, ph);
    pyramid_cache_->temp_texture = create_tex(pw, ph);

    int curr_w = pw;
    int curr_h = ph;

    for (int o = 0; o < kNumOctaves; ++o) {
        pyramid_cache_->octave_sizes.push_back({curr_w, curr_h});
        std::vector<wgpu::Texture> gauss_octave;
        std::vector<wgpu::Texture> dog_octave;

        for (int s = 0; s < kScalesPerOctave + 3; ++s) gauss_octave.push_back(create_tex(curr_w, curr_h));
        for (int s = 0; s < kScalesPerOctave + 2; ++s) dog_octave.push_back(create_tex(curr_w, curr_h));
        
        pyramid_cache_->gaussian_pyramid.push_back(gauss_octave);
        pyramid_cache_->dog_pyramid.push_back(dog_octave);

        curr_w /= 2;
        curr_h /= 2;
    }
}

static volatile bool g_map_done = false;

void SIFTPacked::WriteTimestamp(uint32_t index) {
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassTimestampWrites tsw;
    tsw.querySet = query_set_;
    tsw.beginningOfPassWriteIndex = index;
    tsw.endOfPassWriteIndex = ~0u;
    wgpu::ComputePassDescriptor desc;
    desc.timestampWrites = &tsw;
    wgpu::ComputePassEncoder pass = enc.beginComputePass(desc);
    pass.end();
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
}

void SIFTPacked::PrepareDispatch() {
    // Run a compute shader to calculate workgroup count based on actual keypoint count
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

void SIFTPacked::RunComputeDescriptors() {
    wgpu::CommandEncoder encoder = device_.createCommandEncoder();
    for (int o = 0; o < kNumOctaves; ++o) {
        int w = pyramid_cache_->octave_sizes[o].first;
        int h = pyramid_cache_->octave_sizes[o].second;
        uint32_t params[] = {(uint32_t)w, (uint32_t)h, (uint32_t)o, 0};
        queue_.writeBuffer(buffers_.params16, 0, params, sizeof(params));
        
        wgpu::BindGroupEntry entries[6];
        entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = sizeof(params);
        entries[1].binding = 1; entries[1].buffer = buffers_.keypoints; entries[1].size = buffers_.keypoints.getSize();
        entries[2].binding = 2; entries[2].buffer = buffers_.descriptors; entries[2].size = buffers_.descriptors.getSize();
        entries[3].binding = 3; entries[3].textureView = pyramid_cache_->gaussian_pyramid[o][1].createView();
        entries[4].binding = 4; entries[4].textureView = pyramid_cache_->gaussian_pyramid[o][2].createView();
        entries[5].binding = 5; entries[5].textureView = pyramid_cache_->gaussian_pyramid[o][3].createView();
        
        wgpu::BindGroupDescriptor bind_desc;
        bind_desc.layout = pipeline_descriptor_.getBindGroupLayout(0);
        bind_desc.entryCount = 6;
        bind_desc.entries = entries;
        wgpu::BindGroup bind_group = device_.createBindGroup(bind_desc);
        
        wgpu::ComputePassEncoder pass = encoder.beginComputePass();
        pass.setPipeline(pipeline_descriptor_);
        pass.setBindGroup(0, bind_group, 0, nullptr);
        pass.dispatchWorkgroupsIndirect(buffers_.indirect_dispatch, 12); // Offset 12: descriptor dispatch params
        pass.end();
        wgpu::CommandBuffer commands = encoder.finish();
        queue_.submit(1, &commands);
        encoder = device_.createCommandEncoder();
    }
}

void SIFTPacked::ReadbackDescriptors(std::vector<float>& out_descriptors) {
    size_t count = keypoints_.size();
    if (count == 0) {
        out_descriptors.clear();
        return;
    }
    size_t desc_elem_size = options_.quantizeDescriptors ? 32 * 4 : 128 * 4;
    size_t size = count * desc_elem_size;
    wgpu::Buffer read_buf = createBuffer(size, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    wgpu::CommandEncoder encoder = device_.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers_.descriptors, 0, read_buf, 0, size);
    wgpu::CommandBuffer cmd = encoder.finish();
    queue_.submit(1, &cmd);
    
    bool done = false;
    wgpu::BufferMapCallbackInfo callbackInfo = {};
    callbackInfo.mode = wgpu::CallbackMode::AllowSpontaneous;
    callbackInfo.userdata1 = &done;
    callbackInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* user1, void*) {
        *(bool*)user1 = true;
    };
    read_buf.mapAsync(wgpu::MapMode::Read, 0, size, callbackInfo);
    while (!done) device_.poll(false, nullptr);
    
    out_descriptors.resize(count * 128);
    if (options_.quantizeDescriptors) {
        const uint8_t* data = (const uint8_t*)read_buf.getConstMappedRange(0, size);
        for (size_t i = 0; i < count * 128; ++i) out_descriptors[i] = (float)data[i];
    } else {
        const float* data = (const float*)read_buf.getConstMappedRange(0, size);
        std::memcpy(out_descriptors.data(), data, size);
    }
    read_buf.unmap();
}

void SIFTPacked::DetectKeypoints(const uint8_t* image_data, int width, int height) {
    profiling_ = {}; 
    width_ = width;
    height_ = height;
    keypoints_.clear();

    wgpu::TextureDescriptor tdesc;
    tdesc.size = { (uint32_t)width, (uint32_t)height, 1 };
    tdesc.sampleCount = 1;
    tdesc.mipLevelCount = 1;
    tdesc.format = wgpu::TextureFormat::RGBA8Unorm;
    tdesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment;
    input_texture_ = device_.createTexture(tdesc);

    wgpu::TexelCopyTextureInfo dst;
    dst.texture = input_texture_;
    wgpu::TexelCopyBufferLayout layout;
    layout.offset = 0;
    layout.bytesPerRow = width * 4;
    layout.rowsPerImage = height;
    wgpu::Extent3D extent = { (uint32_t)width, (uint32_t)height, 1 };
    queue_.writeTexture(dst, image_data, width * height * 4, layout, extent);

    Resize(width, height);
    uint32_t zero = 0;
    queue_.writeBuffer(buffers_.keypoints, 0, &zero, 4);

    WriteTimestamp(0);
    int pw = (width + 1) / 2;
    int ph = (height + 1) / 2;

    RunGrayscale(pyramid_cache_->base_texture, pw, ph);
    WriteTimestamp(1);
    BuildPyramids(pw, ph);
    WriteTimestamp(2);
    DetectExtrema();
    PrepareDispatch(); // Prepare indirect dispatch buffer based on actual keypoint count
    WriteTimestamp(3);
    ComputeOrientations();
    WriteTimestamp(4);
    RunComputeDescriptors();
    WriteTimestamp(5);
    WriteTimestamp(6);
    ReadbackKeypoints();
}

void SIFTPacked::ReadbackKeypoints() {
    wgpu::Buffer count_buf = createBuffer(4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    enc.copyBufferToBuffer(buffers_.keypoints, 0, count_buf, 0, 4);
    enc.resolveQuerySet(query_set_, 0, 7, query_resolve_buf_, 0);
    enc.copyBufferToBuffer(query_resolve_buf_, 0, query_result_buf_, 0, 7 * 8);
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);

    g_map_done = false;
    wgpu::BufferMapCallbackInfo callbackInfo = {};
    callbackInfo.mode = wgpu::CallbackMode::AllowSpontaneous;
    callbackInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) { g_map_done = true; };
    count_buf.mapAsync(wgpu::MapMode::Read, 0, 4, callbackInfo);
    while (!g_map_done) device_.poll(false, nullptr);
    const uint32_t* mapped_count = (const uint32_t*)count_buf.getConstMappedRange(0, 4);
    uint32_t count = *mapped_count;
    count_buf.unmap();

    g_map_done = false;
    query_result_buf_.mapAsync(wgpu::MapMode::Read, 0, 7 * 8, callbackInfo);
    while(!g_map_done) device_.poll(false, nullptr);
    const uint64_t* timestamps = (const uint64_t*)query_result_buf_.getConstMappedRange(0, 7 * 8);
    double ns_to_ms = 1e-6 * timestamp_period_;
    profiling_.grayscale_ms = (timestamps[1] - timestamps[0]) * ns_to_ms;
    profiling_.pyramids_ms = (timestamps[2] - timestamps[1]) * ns_to_ms;
    profiling_.extrema_ms = (timestamps[3] - timestamps[2]) * ns_to_ms;
    profiling_.orientation_ms = (timestamps[4] - timestamps[3]) * ns_to_ms;
    profiling_.descriptor_ms = (timestamps[5] - timestamps[4]) * ns_to_ms;
    
    // Total GPU time
    uint64_t gpu_total_diff = timestamps[6] - timestamps[0];
    if (timestamps[6] < timestamps[0]) gpu_total_diff = 0; // Handle wraps/errors
    profiling_.total_ms = gpu_total_diff * ns_to_ms;

    query_result_buf_.unmap();

    if (count == 0) return;
    size_t kp_byte_size = count * 32;
    wgpu::Buffer read_buf = createBuffer(kp_byte_size, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    enc = device_.createCommandEncoder();
    enc.copyBufferToBuffer(buffers_.keypoints, 16, read_buf, 0, kp_byte_size);
    cmd = enc.finish();
    queue_.submit(1, &cmd);
    g_map_done = false;
    read_buf.mapAsync(wgpu::MapMode::Read, 0, kp_byte_size, callbackInfo);
    while (!g_map_done) device_.poll(false, nullptr);
    const float* mapped_data = (const float*)read_buf.getConstMappedRange(0, kp_byte_size);
    for (uint32_t i = 0; i < count; ++i) {
        Keypoint kp;
        kp.x = mapped_data[i * 8 + 0]; kp.y = mapped_data[i * 8 + 1];
        kp.octave = mapped_data[i * 8 + 2]; kp.scale = mapped_data[i * 8 + 3];
        kp.sigma = mapped_data[i * 8 + 4]; kp.orientation = mapped_data[i * 8 + 5];
        keypoints_.push_back(kp);
    }
    read_buf.unmap();
}

void SIFTPacked::RunGrayscale(wgpu::Texture output_tex, int pw, int ph) {
    wgpu::BindGroupEntry entries[2];
    entries[0].binding = 0; entries[0].textureView = input_texture_.createView();
    entries[1].binding = 1; entries[1].textureView = output_tex.createView();
    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_grayscale_.getBindGroupLayout(0);
    desc.entryCount = 2; desc.entries = entries;
    wgpu::BindGroup bind = device_.createBindGroup(desc);
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipeline_grayscale_);
    pass.setBindGroup(0,  bind, 0, nullptr);
    pass.dispatchWorkgroups((pw + 15) / 16,  (ph + 15) / 16, 1);
    pass.end();
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
}

void SIFTPacked::BuildPyramids(int pw, int ph) {
    int w = pw; int h = ph;
    for (int o = 0; o < kNumOctaves; ++o) {
        auto& gauss_octave = pyramid_cache_->gaussian_pyramid[o];
        auto& dog_octave = pyramid_cache_->dog_pyramid[o];
        if (o == 0) RunBlur(pyramid_cache_->base_texture, gauss_octave[0], pyramid_cache_->temp_texture, w, h, kSigmaBase);
        else {
             int prev_w = pyramid_cache_->octave_sizes[o-1].first;
             int prev_h = pyramid_cache_->octave_sizes[o-1].second;
             wgpu::Texture prev_tex = pyramid_cache_->gaussian_pyramid[o-1][kScalesPerOctave];
             RunDownsample(prev_tex, gauss_octave[0], prev_w, prev_h, w, h);
        }
        for (size_t s = 1; s < gauss_octave.size(); ++s) {
            float sigma = GetSigma(s);
            float prev_sigma = GetSigma(s - 1);
            float diff = std::sqrt(sigma * sigma - prev_sigma * prev_sigma);
            RunBlur(gauss_octave[s - 1], gauss_octave[s], pyramid_cache_->temp_texture, w, h, diff);
        }
        for (size_t s = 0; s < dog_octave.size(); ++s) RunDoG(gauss_octave[s], gauss_octave[s + 1], dog_octave[s], w, h);
        w /= 2; h /= 2;
    }
}


void SIFTPacked::RunBlur(wgpu::Texture in_tex, wgpu::Texture out_tex, wgpu::Texture temp_tex, int w, int h, float sigma) {
    int radius = std::ceil(sigma * 3);
    wgpu::Buffer kernel_buf = GetKernelBuffer(sigma, radius);
    uint32_t params[] = { (uint32_t)w, (uint32_t)h, (uint32_t)radius, 0 };
    queue_.writeBuffer(buffers_.params16, 0, params, sizeof(params));
    auto run_pass = [&](wgpu::ComputePipeline pipeline, wgpu::TextureView in_view, wgpu::TextureView out_view) {
        wgpu::BindGroupEntry entries[4];
        entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = 16;
        entries[1].binding = 1; entries[1].textureView = in_view;
        entries[2].binding = 2; entries[2].textureView = out_view;
        entries[3].binding = 3; entries[3].buffer = kernel_buf; entries[3].size = WGPU_WHOLE_SIZE;
        wgpu::BindGroupDescriptor desc;
        desc.layout = pipeline.getBindGroupLayout(0);
        desc.entryCount = 4; desc.entries = entries;
        wgpu::BindGroup bind = device_.createBindGroup(desc);
        wgpu::CommandEncoder enc = device_.createCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0,  bind, 0, nullptr);
        pass.dispatchWorkgroups((w + 15) / 16,  (h + 15) / 16, 1);
        pass.end();
        wgpu::CommandBuffer cmd = enc.finish();
        queue_.submit(1, &cmd);
    };
    run_pass(pipeline_blur_h_, in_tex.createView(), temp_tex.createView());
    run_pass(pipeline_blur_v_, temp_tex.createView(), out_tex.createView());
}

void SIFTPacked::RunDownsample(wgpu::Texture in_tex, wgpu::Texture out_tex, int sw, int sh, int dw, int dh) {
    uint32_t params[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)dw, (uint32_t)dh };
    queue_.writeBuffer(buffers_.params16, 0, params, sizeof(params));
    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0; entries[0].buffer = buffers_.params16; entries[0].size = 16;
    entries[1].binding = 1; entries[1].textureView = in_tex.createView();
    entries[2].binding = 2; entries[2].textureView = out_tex.createView();
    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_downsample_.getBindGroupLayout(0);
    desc.entryCount = 3; desc.entries = entries;
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

void SIFTPacked::RunDoG(wgpu::Texture tex_a, wgpu::Texture tex_b, wgpu::Texture out_tex, int w, int h) {
    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0; entries[0].textureView = tex_a.createView();
    entries[1].binding = 1; entries[1].textureView = tex_b.createView();
    entries[2].binding = 2; entries[2].textureView = out_tex.createView();
    wgpu::BindGroupDescriptor desc;
    desc.layout = pipeline_dog_.getBindGroupLayout(0);
    desc.entryCount = 3; desc.entries = entries;
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

void SIFTPacked::DetectExtrema() {
    for (int o = 0; o < kNumOctaves; ++o) {
        int w = pyramid_cache_->octave_sizes[o].first;
        int h = pyramid_cache_->octave_sizes[o].second;
        for (int s = 1; s <= kScalesPerOctave; ++s) {
            struct { int w, h, o, s; float contrast, edge; } p;
            p.w = w; p.h = h; p.o = o; p.s = s;
            p.contrast = (options_.contrastThreshold / kScalesPerOctave);
            p.edge = options_.edgeThreshold;
            queue_.writeBuffer(buffers_.params_extrema, 0, &p, sizeof(p));
            wgpu::BindGroupEntry entries[5];
            entries[0].binding = 0; entries[0].buffer = buffers_.params_extrema; entries[0].size = 24;
            entries[1].binding = 1; entries[1].textureView = pyramid_cache_->dog_pyramid[o][s - 1].createView();
            entries[2].binding = 2; entries[2].textureView = pyramid_cache_->dog_pyramid[o][s].createView();
            entries[3].binding = 3; entries[3].textureView = pyramid_cache_->dog_pyramid[o][s + 1].createView();
            entries[4].binding = 4; entries[4].buffer = buffers_.keypoints; entries[4].size = 16 + kMaxKeypoints * 32;
            wgpu::BindGroupDescriptor desc;
            desc.layout = pipeline_extrema_.getBindGroupLayout(0);
            desc.entryCount = 5; desc.entries = entries;
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
    }
}

void SIFTPacked::ComputeOrientations() {
    for (int o = 0; o < kNumOctaves; ++o) {
        int w = pyramid_cache_->octave_sizes[o].first;
        int h = pyramid_cache_->octave_sizes[o].second;
        struct { int w, h, o; } p = { w, h, o };
        wgpu::Buffer pbuf = createBuffer(12, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        queue_.writeBuffer(pbuf, 0, &p, sizeof(p));
        wgpu::BindGroupEntry entries[5];
        entries[0].binding = 0; entries[0].buffer = pbuf; entries[0].size = 12;
        entries[1].binding = 1; entries[1].buffer = buffers_.keypoints; entries[1].size = 16 + kMaxKeypoints * 32;
        entries[2].binding = 2; entries[2].textureView = pyramid_cache_->gaussian_pyramid[o][1].createView();
        entries[3].binding = 3; entries[3].textureView = pyramid_cache_->gaussian_pyramid[o][2].createView();
        entries[4].binding = 4; entries[4].textureView = pyramid_cache_->gaussian_pyramid[o][3].createView();
        wgpu::BindGroupDescriptor desc;
        desc.layout = pipeline_orientation_.getBindGroupLayout(0);
        desc.entryCount = 5; desc.entries = entries;
        wgpu::BindGroup bind = device_.createBindGroup(desc);
        wgpu::CommandEncoder enc = device_.createCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.beginComputePass();
        pass.setPipeline(pipeline_orientation_);
        pass.setBindGroup(0,  bind, 0, nullptr);
        pass.dispatchWorkgroupsIndirect(buffers_.indirect_dispatch, 0); // Offset 0: orientation dispatch params
        pass.end();
        wgpu::CommandBuffer cmd = enc.finish();
        queue_.submit(1, &cmd);
    }
}
