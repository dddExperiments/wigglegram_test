struct Params {
    countA: u32,
    countB: u32,
    pad1: u32,
    pad2: u32
}

#include "../common/structs.wgsl"
#include "../common/constants.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> descriptorsA: array<u32>;
@group(0) @binding(2) var<storage, read> descriptorsB: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<MatchResult>;

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE: u32 = 64u;

// 64 threads per workgroup balances parallelism and register pressure for descriptor matching.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idxA = gid.x;
    if (idxA >= params.countA) { return; }

    var bestDistSq = 1e30; 
    var secondDistSq = 1e30;
    var bestIdx = -1;

    for (var i = 0u; i < params.countB; i++) {
        var distSq = 0.0;
        
        for (var k = 0u; k < DESC_DIM / 4u; k++) {
            let valA = descriptorsA[idxA * (DESC_DIM / 4u) + k];
            let valB = descriptorsB[i * (DESC_DIM / 4u) + k];
            
            // Unpack 4 bytes manually
            let a1 = f32(valA & 0xFFu);
            let a2 = f32((valA >> 8u) & 0xFFu);
            let a3 = f32((valA >> 16u) & 0xFFu);
            let a4 = f32((valA >> 24u) & 0xFFu);
            
            let b1 = f32(valB & 0xFFu);
            let b2 = f32((valB >> 8u) & 0xFFu);
            let b3 = f32((valB >> 16u) & 0xFFu);
            let b4 = f32((valB >> 24u) & 0xFFu);
            
            let d1 = a1 - b1;
            let d2 = a2 - b2;
            let d3 = a3 - b3;
            let d4 = a4 - b4;
            
            distSq += d1*d1 + d2*d2 + d3*d3 + d4*d4;
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
