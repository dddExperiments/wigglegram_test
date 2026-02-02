struct Params {
    countA: u32,
    countB: u32,
    threshold: f32,
    pad: u32,
    col0: vec4f,
    col1: vec4f,
    col2: vec4f,
}

#include "../common/structs.wgsl"
#include "../common/constants.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> descriptorsA: array<f32>;
@group(0) @binding(2) var<storage, read> descriptorsB: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<MatchResult>;
@group(0) @binding(4) var<storage, read> keypointsA: array<vec2f>;
@group(0) @binding(5) var<storage, read> keypointsB: array<vec2f>;

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE: u32 = 64u;

// 64 threads per workgroup balances parallelism and register pressure for descriptor matching.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idxA = gid.x;
    if (idxA >= params.countA) { return; }

    let pA = vec3f(keypointsA[idxA], 1.0);
    // Epipolar line in image B: l = F * pA
    // Since we pass F columns: l = col0*pA.x + col1*pA.y + col2*pA.z
    let lineB = params.col0.xyz * pA.x + params.col1.xyz * pA.y + params.col2.xyz * pA.z;
    
    // Pre-calculate line norm for distance
    let lineNorm = sqrt(lineB.x * lineB.x + lineB.y * lineB.y);

    var bestDistSq = 1e30; 
    var secondDistSq = 1e30;
    var bestIdx = -1;

    for (var i = 0u; i < params.countB; i++) {
        let pB = keypointsB[i];
        
        // Epipolar distance: |lineB * pB| / lineNorm
        let distToLine = abs(lineB.x * pB.x + lineB.y * pB.y + lineB.z) / (lineNorm + 1e-6);
        
        if (distToLine > params.threshold) { continue; }

        var distSq = 0.0;
        for (var k = 0u; k < DESC_DIM; k++) {
            let diff = descriptorsA[idxA * DESC_DIM + k] - descriptorsB[i * DESC_DIM + k];
            distSq += diff * diff;
        }

        if (distSq < bestDistSq) {
            secondDistSq = bestDistSq;
            bestDistSq = distSq;
            bestIdx = i32(i);
        } else if (distSq < secondDistSq) {
            secondDistSq = distSq;
        }
    }

    results[idxA].bestIdx = bestIdx;
    results[idxA].bestDistSq = bestDistSq;
    results[idxA].secondDistSq = secondDistSq;
}
