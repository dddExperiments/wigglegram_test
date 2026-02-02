#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <webgpu/webgpu.hpp>

#include "sift_matcher.h"
#include "utils.h"

// Minimal WGPU Init
wgpu::Device CreateDevice() {
    wgpu::Instance instance = wgpu::createInstance();
    if (!instance) { std::cerr << "Failed to create WGPU Instance" << std::endl; exit(1); }
    wgpu::RequestAdapterOptions opts = {};
    opts.powerPreference = wgpu::PowerPreference::HighPerformance;
    wgpu::Adapter adapter = instance.requestAdapter(opts);
    if (!adapter) { std::cerr << "Failed to request Adapter" << std::endl; exit(1); }
    wgpu::DeviceDescriptor desc = {};
    return adapter.requestDevice(desc);
}

void FillRandom(std::vector<float>& d, int size) {
    for(int i=0; i<size; ++i) d[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv) {
    wgpu::Device device = CreateDevice();
    SIFTMatcher matcher;
    matcher.Init(device);

    std::cout << "Running Synthetic Guided Matcher Test..." << std::endl;

    // Create 2 fake features in A, 2 in B.
    // Feature 0: Valid geometric match (Same Y)
    // Feature 1: Invalid geometric match (Diff Y)
    
    // Descriptors: Make them match exactly 0->0, 1->1
    std::vector<float> descA(2 * 128);
    std::vector<float> descB(2 * 128);
    
    // Pair 0 matches
    for(int i=0; i<128; ++i) { descA[i] = 1.0f; descB[i] = 1.0f; }
    // Pair 1 matches
    for(int i=0; i<128; ++i) { descA[128+i] = 0.5f; descB[128+i] = 0.5f; }
    
    // Keypoints
    std::vector<float> kpA = {
        100.0f, 100.0f, // Pt 0
        200.0f, 200.0f  // Pt 1
    };
    std::vector<float> kpB = {
        150.0f, 100.0f, // Pt 0' (shifted X, same Y)
        250.0f, 220.0f  // Pt 1' (shifted X, Diff Y)
    };
    
    // Fundamental Matrix for pure X translation: Enforces y' = y.
    // F = [0  0  0]
    //     [0  0 -1]
    //     [0  1  0]
    // l = F * pA. pA=(x,y,1). 
    // l = [0, -1, y].
    // dot(l, pB) = 0*x' + (-1)*y' + y*1 = y - y'.
    // Distance = |y - y'| / sqrt(0^2 + 1^2) = |y - y'|.
    std::vector<float> F = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 0.0f
    };
    
    // Threshold 5.0 pixels.
    // Pair 0: |100 - 100| = 0. OK.
    // Pair 1: |200 - 220| = 20. > 5. Should be filtered.
    
    std::cout << "Testing MatchGuided..." << std::endl;
    auto matches = matcher.MatchGuided(descA, kpA, descB, kpB, F, 5.0f, 0.9f);
    
    std::cout << "Matches Found: " << matches.size() << std::endl;
    
    bool passed = true;
    if (matches.size() != 1) {
        std::cout << "FAIL: Expected exactly 1 match." << std::endl;
        passed = false;
    } else {
        if (matches[0].queryIdx == 0 && matches[0].trainIdx == 0) {
            std::cout << "SUCCESS: Matched Pair 0 (Geometrically Valid)." << std::endl;
        } else {
            std::cout << "FAIL: Matched wrong pair: " << matches[0].queryIdx << "->" << matches[0].trainIdx << std::endl;
            passed = false;
        }
    }
    
    // Double check: if we allow large threshold, do we get both?
    std::cout << "\nTesting Wide Threshold..." << std::endl;
    auto matches2 = matcher.MatchGuided(descA, kpA, descB, kpB, F, 30.0f, 0.9f);
    if (matches2.size() == 2) {
        std::cout << "SUCCESS: Both matched with wide threshold." << std::endl;
    } else {
        std::cout << "FAIL: Expected 2 matches, got " << matches2.size() << std::endl;
        passed = false;
    }

    return passed ? 0 : 1;
}
