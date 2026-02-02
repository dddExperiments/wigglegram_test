#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "sift_packed.h"
#include "sift_matcher.h"
#include <webgpu/webgpu.hpp>
#include <iostream>

namespace py = pybind11;

class PySIFT {
public:
    PySIFT() {
        // Initialize WebGPU Device
        // Note: In a real Python library, we might want to expose Device creation separately
        // to allow reuse. For now, we create a device per SIFT instance for simplicity,
        // or share a static instance if feasible. SIFTPacked manages its own state but needs a device.
        
        // We borrow the device creation logic from test_main or similar
        instance_ = wgpu::createInstance();
        
        // Request Adapter
        // Synchronous requestAdapter is not standard in JS but wgpu-native C++ wrapper usually provides it or we wait.
        // The C++ wrapper 'instance.requestAdapter' returns the adapter immediately in wgpu-native if available? 
        // Actually wgpu-native/webgpu-cpp might use callbacks or specific helpers.
        // Let's rely on the simple blocking behavior if available, or use the callback-based one with a wait.
        // The specific C++ wrapper we use (webgpu-cpp) + wgpu-native usually allows:
        // adapter = instance.requestAdapter(...)
        
        wgpu::RequestAdapterOptions options = {};
        // We can hint high performance
        options.powerPreference = wgpu::PowerPreference::HighPerformance;
        
        adapter_ = instance_.requestAdapter(options);
        
        // Request Device
        wgpu::DeviceDescriptor deviceDesc = {};
        wgpu::FeatureName requiredFeatures[] = { wgpu::FeatureName::TimestampQuery };
        deviceDesc.requiredFeatures = reinterpret_cast<const WGPUFeatureName*>(requiredFeatures);
        deviceDesc.requiredFeatureCount = 1;
        
        // Toggles
        deviceDesc.requiredLimits = nullptr;
        
        device_ = adapter_.requestDevice(deviceDesc);
        
        // Initialize SIFT
        sift_ = std::make_unique<SIFTPacked>();
        sift_->Init(device_);
    }

    std::vector<std::map<std::string, py::object>> Detect(py::array_t<uint8_t> image) {
        // Input check
        py::buffer_info buf = image.request();
        if (buf.ndim != 2 && buf.ndim != 3) {
            throw std::runtime_error("Number of dimensions must be 2 (grayscale) or 3 (RGBA/RGB)");
        }
        
        int h = buf.shape[0];
        int w = buf.shape[1];
        int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
        
        // SIFT expects RGBA usually if we look at TestPattern? 
        // TestPattern was RGBA.
        // sift_packed.cpp likely uses a compute shader that expects a specific format.
        // Let's check SIFTPacked::DetectKeypoints signature.
        // It takes uint8_t* image_data.
        // And inside, it likely uploads to a texture.
        // In c++/src/sift_packed.cpp, it creates a texture.
        // Default format in webgpu_impl often RGBA8Unorm.
        
        // We will assume input is RGBA for now, or convert if needed.
        // If numpy gives us grayscale, we should expand to RGBA or support grayscale input in SIFT.
        // Let's enforce RGBA for this wrapper version 1.
        
        if (channels != 4) {
             throw std::runtime_error("Image must be RGBA (4 channels)");
        }
        
        const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
        
        // Detect
        sift_->DetectKeypoints(ptr, w, h);
        
        // Readback Descriptors
        std::vector<float> descriptors;
        sift_->ReadbackDescriptors(descriptors);

        // Convert Keypoints to Python compatible format
        const auto& kps = sift_->GetKeypoints();
        std::vector<std::map<std::string, py::object>> result;
        result.reserve(kps.size());
        
        for (size_t i = 0; i < kps.size(); ++i) {
            const auto& kp = kps[i];
            std::map<std::string, py::object> d;
            d["x"] = py::float_(kp.x);
            d["y"] = py::float_(kp.y);
            d["scale"] = py::float_(kp.scale);
            d["orientation"] = py::float_(kp.orientation);
            d["octave"] = py::float_(kp.octave);
            d["sigma"] = py::float_(kp.sigma);
            
            // Slice descriptor
            // Create a numpy array copy or list for now
            // List is safer/easier
            // We can return py::array referencing the specific slice but copy is safer.
            // Let's return a list for now, it's easy to convert to numpy in python if needed.
            // Or better: pybind11 might auto-convert std::vector<float> to list if we passed it.
            // Here we construct manually.
            if (!descriptors.empty()) {
                // Slice
                size_t start = i * 128;
                if (start + 128 <= descriptors.size()) {
                    std::vector<float> desc_vec(descriptors.begin() + start, descriptors.begin() + start + 128);
                    // d["descriptor"] = py::cast(desc_vec); // requires pybind11/stl.h
                    // Direct numpy array
                    d["descriptor"] = py::array(128, desc_vec.data());
                }
            }
            
            result.push_back(d);
        }
        
        return result;
    }

private:
    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    std::unique_ptr<SIFTPacked> sift_;
};

PYBIND11_MODULE(websiftgpu_py, m) {
    m.doc() = "WebGPU SIFT Python Plugin";
    
    py::class_<PySIFT>(m, "SIFT")
        .def(py::init<>())
        .def("detect", &PySIFT::Detect, "Detect keypoints in an RGBA image");

    // --------------- SIFT Matcher ----------------
    py::class_<SIFTMatcher>(m, "SIFTMatcher")
        .def(py::init<>())
        .def("init", [](SIFTMatcher& self) {
            // In a real scenario, we might want to share the device. 
            // However, SIFTMatcher::Init takes a device.
            // PySIFT creates its own device. 
            // We need a way to get a device. 
            // For now, let's create a new device inside SIFTMatcher wrapper or allow passing one?
            // But we can't easily pass wgpu::Device across pybind unless we wrap it.
            
            // Simpler approach for this task: allow Matcher to create its own device internally 
            // OR expose a helper in PySIFTMatcher to create one.
            
            // Actually, looking at SIFTMatcher.Init(device), it expects a device.
            // Let's modify the binding to create the device internally if we can't share.
            // Borrowing device creation logic:
            
            wgpu::Instance instance = wgpu::createInstance();
            wgpu::RequestAdapterOptions options = {};
            options.powerPreference = wgpu::PowerPreference::HighPerformance;
            wgpu::Adapter adapter = instance.requestAdapter(options);
            
            wgpu::DeviceDescriptor deviceDesc = {};
            wgpu::FeatureName requiredFeatures[] = { wgpu::FeatureName::TimestampQuery };
            deviceDesc.requiredFeatures = reinterpret_cast<const WGPUFeatureName*>(requiredFeatures);
            deviceDesc.requiredFeatureCount = 1;
            
            wgpu::Device device = adapter.requestDevice(deviceDesc);
            
            self.Init(device);
        }, "Initialize the matcher with a new WebGPU device")
        
        .def("match", [](SIFTMatcher& self, 
                         py::array_t<float> descA, 
                         py::array_t<float> descB, 
                         float ratio) {
            
            // Convert numpy to std::vector
            py::buffer_info bufA = descA.request();
            py::buffer_info bufB = descB.request();

            if (bufA.ndim != 2 || bufA.shape[1] != 128) throw std::runtime_error("descA must be N x 128");
            if (bufB.ndim != 2 || bufB.shape[1] != 128) throw std::runtime_error("descB must be M x 128");

            // Flat copy
            std::vector<float> vecA((float*)bufA.ptr, (float*)bufA.ptr + bufA.size);
            std::vector<float> vecB((float*)bufB.ptr, (float*)bufB.ptr + bufB.size);

            auto matches = self.MatchDescriptors(vecA, vecB, ratio);
            
            // Return N x 2 numpy array (queryIdx, trainIdx)
            // SIFTMatcher::Match returns struct { trainIdx, queryIdx, distance }
            // process_folder expected pairs.
            
            // Actually, let's return a clean list of lists or numpy array.
            // Numpy array is best for performance.
            
            py::array_t<int> result({ (int)matches.size(), 2 });
            auto r = result.mutable_unchecked<2>();
            
            for (int i = 0; i < matches.size(); i++) {
                r(i, 0) = matches[i].queryIdx;
                r(i, 1) = matches[i].trainIdx;
            }
            
            return result;
        }, "Match descriptors. Returns Nx2 array of [queryIdx, trainIdx]", 
           py::arg("desc1"), py::arg("desc2"), py::arg("ratio") = 0.75f);
}
