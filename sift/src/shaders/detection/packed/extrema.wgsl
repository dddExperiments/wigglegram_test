#include "../../common/structs.wgsl"
#include "../../common/constants.wgsl"

struct Params { width: i32, height: i32, octave: i32, scale: i32, threshold: f32, edgeThreshold: f32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var prevTex: texture_2d<f32>; // Packed
@group(0) @binding(2) var currTex: texture_2d<f32>; // Packed
@group(0) @binding(3) var nextTex: texture_2d<f32>; // Packed
@group(0) @binding(4) var<storage, read_write> keypoints: KeypointList;

// Helper to Sample Logical Pixel (lx, ly)
fn getVal(tex: texture_2d<f32>, lx: i32, ly: i32) -> f32 {
    let px = lx / 2;
    let py = ly / 2;
    let mx = lx % 2;
    let my = ly % 2;
    let val = textureLoad(tex, vec2i(px, py), 0);
    
    // Select component
    if (my == 0) {
        return select(val.x, val.y, mx == 1);
    } else {
        return select(val.z, val.w, mx == 1);
    }
}

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    workgroupBarrier();

    // gid is PACKED coordinates. 
    // We process 4 logical pixels per thread? Or 1 logical pixel per thread?
    // If we map 1 thread -> 1 packed pixel:
    // It processes logical (2x, 2y), (2x+1, 2y)...
    // This is efficient.
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    
    // Iterate 4 sub-pixels
    for (var sy = 0; sy < 2; sy++) {
        for (var sx = 0; sx < 2; sx++) {
            let lx = px * 2 + sx; // Logical X
            let ly = py * 2 + sy; // Logical Y
            
            // Check logical bounds
            if (lx < 1 || ly < 1 || lx >= params.width * 2 - 1 || ly >= params.height * 2 - 1) { continue; }
            
            checkKeypoint(lx, ly);
        }
    }
    
    // ... Aggregation logic (same as opt) ...
    // Note: wgCount is shared, so multiple `checkKeypoint` calls can increment it.
}

fn checkKeypoint(x: i32, y: i32) {
    // Standard checks using getVal()
    let val = getVal(currTex, x, y);
    if (abs(val) < params.threshold) { return; }
    
    var isMax = true;
    var isMin = true;
    
    // 3x3x3 Neighbor check
    for (var vz = -1; vz <= 1; vz++) {
        for (var vy = -1; vy <= 1; vy++) {
            for (var vx = -1; vx <= 1; vx++) {
                if (vz == 0 && vy == 0 && vx == 0) { continue; }
                var nVal: f32;
                if (vz == -1) { nVal = getVal(prevTex, x+vx, y+vy); }
                else if (vz == 0) { nVal = getVal(currTex, x+vx, y+vy); }
                else { nVal = getVal(nextTex, x+vx, y+vy); }
                
                if (nVal >= val) { isMax = false; }
                if (nVal <= val) { isMin = false; }
            }
        }
    }
    
    if (!isMax && !isMin) { return; }
    
    // Edge Check
    let dxx = getVal(currTex, x+1, y) + getVal(currTex, x-1, y) - 2.0 * val;
    let dyy = getVal(currTex, x, y+1) + getVal(currTex, x, y-1) - 2.0 * val;
    let dxy = (getVal(currTex, x+1, y+1) - getVal(currTex, x+1, y-1) - getVal(currTex, x-1, y+1) + getVal(currTex, x-1, y-1)) * 0.25;
    // Edge check: Reject points that have a large principal curvature in one direction
    // but a small one in the other (edges).
    // Uses the ratio of eigenvalues of the 2x2 Hessian matrix.
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    let r = params.edgeThreshold;
    
    if (det > 0.0 && (tr * tr * r) < ((r + 1.0) * (r + 1.0) * det)) {
        // Add Keypoint
        let globalIdx = atomicAdd(&keypoints.count, 1u);
        let scaleMult = pow(2.0, f32(params.octave));
        
        keypoints.points[globalIdx].x = f32(x) * scaleMult;
        keypoints.points[globalIdx].y = f32(y) * scaleMult;
        keypoints.points[globalIdx].octave = f32(params.octave);
        keypoints.points[globalIdx].scale = f32(params.scale);
        keypoints.points[globalIdx].sigma = SIGMA_BASE * pow(2.0, (f32(params.scale) / SCALES_PER_OCTAVE)) * scaleMult;
        keypoints.points[globalIdx].orientation = 0.0;
    }
}
