struct Params {
    countA: u32,
    countB: u32,
    pad1: u32,
    pad2: u32,
}

struct MatchResult {
    bestIdx: i32,
    bestDist: f32,
    secondDist: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> listA: array<f32>; // 128 floats per desc
@group(0) @binding(2) var<storage, read> listB: array<f32>; // 128 floats per desc
@group(0) @binding(3) var<storage, read_write> results: array<MatchResult>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idxA = gid.x;
    if (idxA >= params.countA) { return; }

    // Load descriptor A
    var descA: array<f32, 128>;
    for (var k = 0u; k < 128u; k++) {
        descA[k] = listA[idxA * 128u + k];
    }

    var bestDist = 1e30;
    var secondDist = 1e30;
    var bestIdx = -1;

    // Brute force search list B
    for (var idxB = 0u; idxB < params.countB; idxB++) {
        var distSq = 0.0;
        for (var k = 0u; k < 128u; k++) {
            let diff = descA[k] - listB[idxB * 128u + k];
            distSq += diff * diff;
        }

        // Update top 2
        if (distSq < bestDist) {
            secondDist = bestDist;
            bestDist = distSq;
            bestIdx = i32(idxB);
        } else if (distSq < secondDist) {
            secondDist = distSq;
        }
    }

    // Write result 
    results[idxA].bestIdx = bestIdx;
    results[idxA].bestDist = bestDist;
    results[idxA].secondDist = secondDist;
}
