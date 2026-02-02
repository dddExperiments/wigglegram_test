struct Params {
    countA: u32,
    countB: u32,
    pad1: u32,
    pad2: u32
}

#include "../common/structs.wgsl"
#include "../common/constants.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> descriptorsA: array<f32>;
@group(0) @binding(2) var<storage, read> descriptorsB: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<MatchResult>;

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE: u32 = 64u;

// 64 threads per workgroup balances parallelism and register pressure for descriptor matching.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idxA = gid.x;
    if (idxA >= params.countA) { return; }

    var bestDistSq = 1e30; // Infinity
    var secondDistSq = 1e30;
    var bestIdx = -1;

    // Loop over all descriptors in set B
    for (var i = 0u; i < params.countB; i++) {
        var distSq = 0.0;
        
        // Compute Euclidean distance squared (128 dimensions)
        // Loop unrolling might help, but let's keep it simple
        for (var k = 0u; k < DESC_DIM; k++) {
            let valA = descriptorsA[idxA * DESC_DIM + k];
            let valB = descriptorsB[i * DESC_DIM + k];
            let diff = valA - valB;
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
