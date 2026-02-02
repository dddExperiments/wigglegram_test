#include "sift_matcher.h"
#include <webgpu/webgpu.h>
#include "embedded_shaders.h"
#include "utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

struct GPUMatchResult {
    int32_t bestIdx;
    float bestDistSq;
    float secondDistSq;
    float pad;
};

SIFTMatcher::SIFTMatcher() = default;
SIFTMatcher::~SIFTMatcher() = default;

void SIFTMatcher::Init(wgpu::Device device) {
    device_ = device;
    queue_ = device.getQueue();

    auto create_pipeline = [&](const std::string& name) {
        std::string code = loadShader(name);
        if (code.empty()) return wgpu::ComputePipeline();
        wgpu::ShaderModuleDescriptor sd = {};
        wgpu::ShaderSourceWGSL wd = {};
        wd.chain.sType = wgpu::SType::ShaderSourceWGSL;
        wd.code = wgpu::StringView(code.c_str());
        sd.nextInChain = &wd.chain;
        wgpu::ShaderModule mod = device_.createShaderModule(sd);
        wgpu::ComputePipelineDescriptor pd = {};
        pd.compute.module = mod;
        pd.compute.entryPoint = wgpu::StringView("main");
        return device_.createComputePipeline(pd);
    };

    pipeline_ = create_pipeline("matcher.wgsl");
    pipeline_quant_ = create_pipeline("matcher_quantized.wgsl");
    pipeline_guided_ = create_pipeline("matcher_guided.wgsl");
}

std::string SIFTMatcher::loadShader(const std::string& name) {
    std::string code = shader_embed::GetShader(name);
    if (!code.empty()) return code;

    std::vector<std::string> search_paths = {
        "../../src/shaders/matching/",
        "../src/shaders/matching/",
        "src/shaders/matching/"
    };

    for (const auto& base : search_paths) {
        try {
            return utils::readFile(base + name);
        } catch (...) {}
    }
    return "";
}

wgpu::Buffer SIFTMatcher::createBuffer(size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc = {};
    desc.size = (size + 3) & ~3u;
    desc.usage = usage;
    return device_.createBuffer(desc);
}

std::vector<Match> SIFTMatcher::MatchDescriptors(const std::vector<float>& descA, 
                                            const std::vector<float>& descB, 
                                            float ratio_threshold,
                                            bool quantize) {
    std::vector<Match> matches;
    if (descA.empty() || descB.empty()) return matches;

    uint32_t countA = descA.size() / 128;
    uint32_t countB = descB.size() / 128;

    wgpu::ComputePipeline pipe = quantize ? pipeline_quant_ : pipeline_;
    if (!pipe) {
        std::cerr << "[SIFTMatcher] Pipeline not initialized" << std::endl;
        return matches;
    }

    size_t sizeA, sizeB;
    wgpu::Buffer bufA, bufB;
    if (quantize) {
        sizeA = countA * 32 * 4;
        sizeB = countB * 32 * 4;
        std::vector<uint32_t> qA(countA * 32), qB(countB * 32);
        for(size_t i=0; i<countA; ++i) {
            for(size_t k=0; k<32; ++k) {
                uint32_t v = 0;
                for(size_t j=0; j<4; ++j) v |= (uint32_t(descA[i*128 + k*4 + j]) & 0xFF) << (j*8);
                qA[i*32 + k] = v;
            }
        }
        for(size_t i=0; i<countB; ++i) {
            for(size_t k=0; k<32; ++k) {
                uint32_t v = 0;
                for(size_t j=0; j<4; ++j) v |= (uint32_t(descB[i*128 + k*4 + j]) & 0xFF) << (j*8);
                qB[i*32 + k] = v;
            }
        }
        bufA = createBuffer(sizeA, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        bufB = createBuffer(sizeB, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        queue_.writeBuffer(bufA, 0, qA.data(), sizeA);
        queue_.writeBuffer(bufB, 0, qB.data(), sizeB);
    } else {
        sizeA = descA.size() * 4;
        sizeB = descB.size() * 4;
        bufA = createBuffer(sizeA, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        bufB = createBuffer(sizeB, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        queue_.writeBuffer(bufA, 0, descA.data(), sizeA);
        queue_.writeBuffer(bufB, 0, descB.data(), sizeB);
    }

    size_t resSize = countA * sizeof(GPUMatchResult);
    wgpu::Buffer bufRes = createBuffer(resSize, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    uint32_t params[] = {countA, countB, 0, 0};
    wgpu::Buffer bufParams = createBuffer(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    queue_.writeBuffer(bufParams, 0, params, 16);

    wgpu::BindGroupEntry entries[4];
    entries[0].binding = 0; entries[0].buffer = bufParams; entries[0].size = 16;
    entries[1].binding = 1; entries[1].buffer = bufA; entries[1].size = sizeA;
    entries[2].binding = 2; entries[2].buffer = bufB; entries[2].size = sizeB;
    entries[3].binding = 3; entries[3].buffer = bufRes; entries[3].size = resSize;
    wgpu::BindGroupDescriptor bgd = {};
    bgd.layout = pipe.getBindGroupLayout(0);
    bgd.entryCount = 4; bgd.entries = entries;
    wgpu::BindGroup bg = device_.createBindGroup(bgd);

    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg, 0, nullptr);
    pass.dispatchWorkgroups((countA + 63) / 64, 1, 1);
    pass.end();
    wgpu::Buffer readBuf = createBuffer(resSize, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    enc.copyBufferToBuffer(bufRes, 0, readBuf, 0, resSize);
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);

    bool done = false;
    wgpu::BufferMapCallbackInfo ci = {};
    ci.mode = wgpu::CallbackMode::AllowSpontaneous; ci.userdata1 = &done;
    ci.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* u1, void*) { *(bool*)u1 = true; };
    readBuf.mapAsync(wgpu::MapMode::Read, 0, resSize, ci);
    while(!done) device_.poll(false, nullptr);
    const GPUMatchResult* gpuRes = (const GPUMatchResult*)readBuf.getConstMappedRange(0, resSize);
    float ratioSq = ratio_threshold * ratio_threshold;
    for (uint32_t i = 0; i < countA; ++i) {
        if (gpuRes[i].bestIdx >= 0 && gpuRes[i].bestDistSq < ratioSq * gpuRes[i].secondDistSq) {
            Match m; m.queryIdx = i; m.trainIdx = gpuRes[i].bestIdx; m.distance = sqrt(gpuRes[i].bestDistSq);
            matches.push_back(m);
        }
    }
    readBuf.unmap();
    return matches;
}

std::vector<Match> SIFTMatcher::MatchGuided(const std::vector<float>& descA, const std::vector<float>& kpsA,
                                            const std::vector<float>& descB, const std::vector<float>& kpsB,
                                            const std::vector<float>& F,
                                            float threshold, float ratio_threshold) {
    std::vector<Match> matches;
    if (descA.empty() || descB.empty() || !pipeline_guided_) return matches;
    uint32_t countA = descA.size() / 128;
    uint32_t countB = descB.size() / 128;
    size_t sizeA = descA.size() * 4, sizeB = descB.size() * 4;
    size_t sizeKpA = kpsA.size() * 4, sizeKpB = kpsB.size() * 4;
    wgpu::Buffer bDA = createBuffer(sizeA, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    wgpu::Buffer bDB = createBuffer(sizeB, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    wgpu::Buffer bKA = createBuffer(sizeKpA, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    wgpu::Buffer bKB = createBuffer(sizeKpB, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    queue_.writeBuffer(bDA, 0, descA.data(), sizeA); queue_.writeBuffer(bDB, 0, descB.data(), sizeB);
    queue_.writeBuffer(bKA, 0, kpsA.data(), sizeKpA); queue_.writeBuffer(bKB, 0, kpsB.data(), sizeKpB);
    size_t resSize = countA * sizeof(GPUMatchResult);
    wgpu::Buffer bR = createBuffer(resSize, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    struct { uint32_t cA, cB; float t; uint32_t p; float col0[4], col1[4], col2[4]; } p;
    p.cA = countA; p.cB = countB; p.t = threshold; p.p = 0;
    p.col0[0] = F[0]; p.col0[1] = F[3]; p.col0[2] = F[6]; p.col0[3] = 0;
    p.col1[0] = F[1]; p.col1[1] = F[4]; p.col1[2] = F[7]; p.col1[3] = 0;
    p.col2[0] = F[2]; p.col2[1] = F[5]; p.col2[2] = F[8]; p.col2[3] = 0;
    wgpu::Buffer bP = createBuffer(sizeof(p), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    queue_.writeBuffer(bP, 0, &p, sizeof(p));
    wgpu::BindGroupEntry e[6];
    e[0].binding = 0; e[0].buffer = bP; e[0].size = sizeof(p);
    e[1].binding = 1; e[1].buffer = bDA; e[1].size = sizeA;
    e[2].binding = 2; e[2].buffer = bDB; e[2].size = sizeB;
    e[3].binding = 3; e[3].buffer = bR; e[3].size = resSize;
    e[4].binding = 4; e[4].buffer = bKA; e[4].size = sizeKpA;
    e[5].binding = 5; e[5].buffer = bKB; e[5].size = sizeKpB;
    wgpu::BindGroupDescriptor bgd = {}; bgd.layout = pipeline_guided_.getBindGroupLayout(0);
    bgd.entryCount = 6; bgd.entries = e;
    wgpu::BindGroup bg = device_.createBindGroup(bgd);
    wgpu::CommandEncoder enc = device_.createCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.beginComputePass();
    pass.setPipeline(pipeline_guided_); pass.setBindGroup(0, bg, 0, nullptr);
    pass.dispatchWorkgroups((countA + 63) / 64, 1, 1);
    pass.end();
    wgpu::Buffer readBuf = createBuffer(resSize, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    enc.copyBufferToBuffer(bR, 0, readBuf, 0, resSize);
    wgpu::CommandBuffer cmd = enc.finish();
    queue_.submit(1, &cmd);
    bool done = false; 
    wgpu::BufferMapCallbackInfo cbi = {}; cbi.mode = wgpu::CallbackMode::AllowSpontaneous; cbi.userdata1 = &done;
    cbi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* u1, void*) { *(bool*)u1 = true; };
    readBuf.mapAsync(wgpu::MapMode::Read, 0, resSize, cbi);
    while(!done) device_.poll(false, nullptr);
    const GPUMatchResult* gpuRes = (const GPUMatchResult*)readBuf.getConstMappedRange(0, resSize);
    float ratioSq = ratio_threshold * ratio_threshold;
    for (uint32_t i = 0; i < countA; ++i) {
        if (gpuRes[i].bestIdx >= 0 && gpuRes[i].bestDistSq < ratioSq * gpuRes[i].secondDistSq) {
            Match m; m.queryIdx = i; m.trainIdx = gpuRes[i].bestIdx; m.distance = sqrt(gpuRes[i].bestDistSq);
            matches.push_back(m);
        }
    }
    readBuf.unmap();
    return matches;
}
