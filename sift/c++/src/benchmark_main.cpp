#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include "absl/time/time.h"
#include "absl/time/clock.h"

#include <webgpu/webgpu.hpp>
#include <webgpu/webgpu.h>
#include "sift_packed.h"
#include "utils.h"

// wgpu-native extension function
extern "C" float wgpuQueueGetTimestampPeriod(WGPUQueue queue);

wgpu::Device CreateDevice() {
    wgpu::Instance instance = wgpu::createInstance();
    wgpu::RequestAdapterOptions options = {};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    wgpu::Adapter adapter = instance.requestAdapter(options);
    wgpu::DeviceDescriptor deviceDesc = {};
    wgpu::FeatureName requiredFeatures[] = { wgpu::FeatureName::TimestampQuery };
    deviceDesc.requiredFeatures = reinterpret_cast<const WGPUFeatureName*>(requiredFeatures);
    deviceDesc.requiredFeatureCount = 1;
    wgpu::Device device = adapter.requestDevice(deviceDesc);
    if (!device) {
        std::cerr << "Failed to create device. TimestampQuery feature might be missing." << std::endl;
        exit(1);
    }
    return device;
}

struct PerfStats {
    std::string name;
    std::vector<double> times;
    void print() {
        if (times.empty()) return;
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();
        std::sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        std::cout << std::left << std::setw(15) << name << ": Mean=" << std::fixed << std::setprecision(2) << mean 
                  << "ms, Median=" << median << "ms, Min=" << times.front() << "ms, Max=" << times.back() << "ms" << std::endl;
    }
};

void RunBenchmark(wgpu::Device device, const std::vector<uint8_t>& image_data, int width, int height, bool quantize) {
    SIFTOptions options;
    options.quantizeDescriptors = quantize;
    SIFTPacked sift;
    sift.Init(device, options);

    std::cout << "\n>>> Benchmarking " << (quantize ? "QUANTIZED" : "FLOAT32") << " (" << width << "x" << height << ")..." << std::endl;
    std::cout << "Warming up..." << std::flush;
    
    // Warmup
    sift.DetectKeypoints(image_data.data(), width, height);
    std::cout << " done" << std::endl;
    
    int iterations = 30;
    std::vector<PerfStats> stats = {
        {"Grayscale", {}}, {"Pyramids", {}}, {"Extrema", {}}, {"Orientation", {}}, {"Descriptor", {}}, {"Download", {}}, {"Total (GPU)", {}}, {"Host Total", {}}
    };

    for (int i = 0; i < iterations; ++i) {
        auto t0 = absl::Now();
        sift.DetectKeypoints(image_data.data(), width, height);
        auto t1 = absl::Now();
        double host_ms = absl::ToDoubleMilliseconds(t1 - t0);

        const auto& prof = sift.GetProfiling();
        stats[0].times.push_back(prof.grayscale_ms);
        stats[1].times.push_back(prof.pyramids_ms);
        stats[2].times.push_back(prof.extrema_ms);
        stats[3].times.push_back(prof.orientation_ms);
        stats[4].times.push_back(prof.descriptor_ms);
        stats[5].times.push_back(prof.download_ms);
        stats[6].times.push_back(prof.total_ms);
        stats[7].times.push_back(host_ms);
    }

    for (auto& s : stats) s.print();
    std::cout << "Detected " << sift.GetKeypointsCount() << " keypoints." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    int width, height;
    std::vector<uint8_t> image_data = utils::loadImage(argv[1], &width, &height);
    if (image_data.empty()) return 1;

    wgpu::Device device = CreateDevice();
    

    RunBenchmark(device, image_data, width, height, false);
    RunBenchmark(device, image_data, width, height, true);

    return 0;
}
