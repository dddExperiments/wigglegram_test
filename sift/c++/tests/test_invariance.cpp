#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <webgpu/webgpu.hpp>

#include "sift_packed.h"
#include "utils.h"

// --- Helper to Create WebGPU Device (Copied/Adapted from main.cpp) ---
wgpu::Device CreateDevice() {
    wgpu::Instance instance = wgpu::createInstance();
    if (!instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        exit(1);
    }
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    wgpu::Adapter adapter = instance.requestAdapter(adapterOptions);
    if (!adapter) {
        std::cerr << "Adapter request failed." << std::endl;
        exit(1);
    }
    wgpu::DeviceDescriptor deviceDesc = {};
    deviceDesc.label = wgpu::StringView("WebSIFTGPU Invariance Test");
    wgpu::Device device = adapter.requestDevice(deviceDesc);
    if (!device) {
        std::cerr << "Device request failed." << std::endl;
        exit(1);
    }
    return device;
}

// --- Utils ---
std::vector<uint8_t> MatToRGBA(const cv::Mat& img) {
    cv::Mat rgba;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, rgba, cv::COLOR_BGRA2RGBA);
    } else {
        cv::cvtColor(img, rgba, cv::COLOR_GRAY2RGBA);
    }
    std::vector<uint8_t> data(rgba.total() * 4);
    std::memcpy(data.data(), rgba.data, data.size());
    return data;
}

void ToOpenCV(const std::vector<Keypoint>& kp_in, const std::vector<float>& desc_in, 
              std::vector<cv::KeyPoint>& kp_out, cv::Mat& desc_out) {
    kp_out.clear();
    kp_out.reserve(kp_in.size());
    
    // Check descriptor size. SIFT uses 128 floats per descriptor.
    size_t num_kp = kp_in.size();
    if (desc_in.size() < num_kp * 128) {
        // If descriptors are empty/partial, handled gracefully?
        // But we expect them to be present.
    }
    
    // Create descriptor mat
    if (!desc_in.empty() && desc_in.size() >= num_kp * 128) {
        desc_out = cv::Mat(num_kp, 128, CV_32F);
    }
    
    for (size_t i = 0; i < num_kp; ++i) {
        const auto& k = kp_in[i];
        kp_out.emplace_back(k.x, k.y, k.scale); 
        kp_out.back().angle = k.orientation * 180.0f / CV_PI;
        kp_out.back().octave = (int)k.octave; 
        
        if (!desc_out.empty()) {
            std::memcpy(desc_out.ptr<float>(i), &desc_in[i * 128], 128 * sizeof(float));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./websiftgpu_invariance <image_path>" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    cv::Mat originalImg = cv::imread(imagePath);
    if (originalImg.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << imagePath << " (" << originalImg.cols << "x" << originalImg.rows << ")" << std::endl;

    // Init WebGPU
    wgpu::Device device = CreateDevice();
    SIFTPacked sift;
    sift.Init(device);

    std::ofstream csvFile("rotation_invariance.csv");
    csvFile << "angle,inliers,matches,kp_original,kp_rotated\n";

    // --- Detect on Original (Angle 0) ---
    std::vector<uint8_t> originalData = MatToRGBA(originalImg);
    sift.DetectKeypoints(originalData.data(), originalImg.cols, originalImg.rows);
    
    const auto& kpOriginalRaw = sift.GetKeypoints();
    std::vector<float> descOriginalRaw;
    sift.ReadbackDescriptors(descOriginalRaw);
    
    std::vector<cv::KeyPoint> kpOriginal;
    cv::Mat descOriginal;
    ToOpenCV(kpOriginalRaw, descOriginalRaw, kpOriginal, descOriginal);
    
    std::cout << "Original Keypoints: " << kpOriginal.size() << std::endl;

    if (kpOriginal.empty()) {
        std::cerr << "No keypoints on original image!" << std::endl;
        return 1;
    }

    // --- Loop ---
    cv::BFMatcher matcher(cv::NORM_L2, true); // Cross check

    for (int angle = 0; angle <= 360; angle += 10) {
        cv::Mat rotatedImg;
        cv::Point2f center(originalImg.cols / 2.0f, originalImg.rows / 2.0f);
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Rect bbox = cv::RotatedRect(center, originalImg.size(), angle).boundingRect();
        rotMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rotMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        
        cv::warpAffine(originalImg, rotatedImg, rotMat, bbox.size());

        std::vector<uint8_t> rotData = MatToRGBA(rotatedImg);
        sift.DetectKeypoints(rotData.data(), rotatedImg.cols, rotatedImg.rows);
        
        const auto& kpRotRaw = sift.GetKeypoints();
        std::vector<float> descRotRaw;
        sift.ReadbackDescriptors(descRotRaw);
        
        std::vector<cv::KeyPoint> kpRotated;
        cv::Mat descRotated;
        ToOpenCV(kpRotRaw, descRotRaw, kpRotated, descRotated);

        int inliers = 0;
        int matchesCount = 0;
        
        if (!kpRotated.empty() && !descRotated.empty() && !descOriginal.empty()) {
            std::vector<cv::DMatch> matches;
            matcher.match(descOriginal, descRotated, matches);
            matchesCount = matches.size();
            
            std::vector<cv::Point2f> pts1, pts2;
            for (const auto& m : matches) {
                pts1.push_back(kpOriginal[m.queryIdx].pt);
                pts2.push_back(kpRotated[m.trainIdx].pt);
            }
            
            if (pts1.size() >= 4) {
                cv::Mat mask;
                cv::findHomography(pts1, pts2, cv::RANSAC, 5.0, mask);
                inliers = cv::countNonZero(mask);
            }
        }
        
        // CSV Output
        csvFile << angle << "," << inliers << "," << matchesCount << "," << kpOriginal.size() << "," << kpRotated.size() << "\n";
        std::cout << "Angle " << angle << ": " << inliers << " inliers / " << matchesCount << " matches" << std::endl;
        
        // Debug first iteration
        if (angle == 0) {
            // Self-match should be perfect
            // std::cout << "Self matching test: " << inliers << std::endl;
        }
    }

    csvFile.close();
    std::cout << "Done. Results saved to rotation_invariance.csv" << std::endl;
    return 0;
}
