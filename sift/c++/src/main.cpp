#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>

// WebGPU API (Standard C++ Header from WebGPU-distribution)
#include <webgpu/webgpu.hpp>

#include "sift_default.h"
#include "sift_packed.h"
#include "utils.h"

// Helper to create device using standard WebGPU C++ API (WebGPU-Cpp wrapper)
wgpu::Device CreateDevice() {
    // 1. Create Instance
    wgpu::Instance instance = wgpu::createInstance();
    if (!instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        exit(1);
    }

    // 2. Request Adapter
    std::cout << "Requesting adapter..." << std::endl;
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    
    // Using synchronous helper from webgpu.hpp
    wgpu::Adapter adapter = instance.requestAdapter(adapterOptions);
    if (!adapter) {
        std::cerr << "Adapter request failed." << std::endl;
        exit(1);
    }
    
    // Print info
    wgpu::AdapterInfo info = {};
    adapter.getInfo(&info);
    std::cout << "Using adapter: " << (info.device.data ? info.device.data : "Unknown") << std::endl;

    // 3. Request Device
    std::cout << "Requesting device..." << std::endl;
    wgpu::DeviceDescriptor deviceDesc = {};
    deviceDesc.label = wgpu::StringView("WebSIFTGPU Device");
    
    // Request TimestampQuery feature
    std::vector<wgpu::FeatureName> requiredFeatures;
    requiredFeatures.push_back(wgpu::FeatureName::TimestampQuery);
    
    deviceDesc.requiredFeatures = (const WGPUFeatureName*) requiredFeatures.data();
    deviceDesc.requiredFeatureCount = requiredFeatures.size();
    
    deviceDesc.requiredLimits = nullptr;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = wgpu::StringView("Default Queue");

    // Error Callback
    deviceDesc.uncapturedErrorCallbackInfo.callback = [](const WGPUDevice* device, WGPUErrorType type, WGPUStringView message, void* user1, void* user2) {
        std::cerr << "Uncaptured WebGPU Error (" << type << "): ";
        if (message.data) std::cerr << std::string(message.data, message.length);
        std::cerr << std::endl;
    };

    // Using synchronous helper from webgpu.hpp
    wgpu::Device device = adapter.requestDevice(deviceDesc);
    if (!device) {
        std::cerr << "Device request failed." << std::endl;
        exit(1);
    }
    
    return device;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [packed|unpacked] [output_json]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string mode = "unpacked";
    if (argc >= 3) {
        mode = argv[2];
    }
    std::string output_path = "";
    if (argc >= 4) {
        output_path = argv[3];
    }

    std::cout << "Loading image: " << image_path << std::endl;
    int width, height;
    std::vector<uint8_t> image_data;
    try {
        image_data = utils::loadImage(image_path, &width, &height);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    std::cout << "Image size: " << width << "x" << height << std::endl;

    std::cout << "Initializing WebGPU..." << std::endl;
    wgpu::Device device = CreateDevice();
    
    std::unique_ptr<SIFTBase> sift;
    if (mode == "packed") {
        std::cout << "Mode: Packed" << std::endl;
        sift = std::make_unique<SIFTPacked>();
    } else {
        std::cout << "Mode: Unpacked (Default)" << std::endl;
        sift = std::make_unique<SIFTDefault>();
    }

    sift->Init(device);

    std::cout << "Detecting keypoints..." << std::endl;
    sift->DetectKeypoints(image_data.data(), width, height);

    const auto& keypoints = sift->GetKeypoints();
    std::cout << "Found " << keypoints.size() << " keypoints." << std::endl;

    // Compute Descriptors (if supported/implemented by calling ReadbackDescriptors)
    // SIFTBase doesn't declare virtual ComputeDescriptors, but SIFTPacked has ReadbackDescriptors.
    // We need to cast or ensure interface. SIFTBase has pure virtual? No.
    // Let's assume SIFTPacked is used mostly.
    
    std::vector<float> descriptors;
    if (mode == "packed") {
        ((SIFTPacked*)sift.get())->ReadbackDescriptors(descriptors);
    } else {
        // Implement for default if needed, or skip
        // ((SIFTDefault*)sift.get())->ReadbackDescriptors(descriptors); // If Implemented
        // For now, only Packed supports descriptors readback in this CLI example 
        // based on previous analysis of files (SIFTDefault wasn't shown fully but likely similar)
    }

    if (!output_path.empty()) {
        std::cout << "Writing results to " << output_path << std::endl;
        std::ofstream outfile(output_path);
        outfile << "{\n";
        outfile << "  \"keypoints\": [\n";
        for (size_t i = 0; i < keypoints.size(); ++i) {
            const auto& kp = keypoints[i];
            outfile << "    { \"x\": " << kp.x << ", \"y\": " << kp.y 
                    << ", \"scale\": " << kp.scale << ", \"orientation\": " << kp.orientation 
                    << ", \"octave\": " << kp.octave << " }";
            if (i < keypoints.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ],\n";
        
        outfile << "  \"descriptors\": [\n";
        int desc_len = 128;
        int num_desc = descriptors.size() / desc_len;
        for (int i = 0; i < num_desc; ++i) {
            outfile << "    [";
            for (int j = 0; j < desc_len; ++j) {
                outfile << descriptors[i*desc_len + j];
                if (j < desc_len - 1) outfile << ",";
            }
            outfile << "]";
            if (i < num_desc - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]\n";
        outfile << "}\n";
        outfile.close();
    }

    return 0;
}
