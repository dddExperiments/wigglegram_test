#include "../../common/structs.wgsl"
#include "../../common/constants.wgsl"

struct Params {
    width: i32, height: i32, octave: i32, scale: i32, threshold: f32, edgeThreshold: f32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var prevTex: texture_2d<f32>;
@group(0) @binding(2) var currTex: texture_2d<f32>;
@group(0) @binding(3) var nextTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> keypoints: KeypointList;

var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgGlobalOffset: u32;

fn getVal(tex: texture_2d<f32>, x: i32, y: i32) -> f32 {
    return textureLoad(tex, vec2i(x, y), 0).r;
}

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    // Init workgroup atomic
    if (lid == 0u) {
        atomicStore(&wgCount, 0u);
    }
    workgroupBarrier();

    let x = i32(gid.x);
    let y = i32(gid.y);
    
    var isFeature = false;
    
    if (x >= 1 && y >= 1 && x < params.width - 1 && y < params.height - 1) {
        let val = getVal(currTex, x, y);
        if (abs(val) >= params.threshold) {
            var isMax = true;
            var isMin = true;
            // Checks...
            for (var vz = -1; vz <= 1; vz++) {
                for (var vy = -1; vy <= 1; vy++) {
                    for (var vx = -1; vx <= 1; vx++) {
                        if (vx == 0 && vy == 0 && vz == 0) { continue; }
                        var neighborVal: f32;
                        if (vz == -1) { neighborVal = getVal(prevTex, x + vx, y + vy); }
                        else if (vz == 0) { neighborVal = getVal(currTex, x + vx, y + vy); }
                        else { neighborVal = getVal(nextTex, x + vx, y + vy); }
                        
                        if (neighborVal >= val) { isMax = false; }
                        if (neighborVal <= val) { isMin = false; }
                        
                        if (!isMax && !isMin) { break; } 
                    }
                    if (!isMax && !isMin) { break; }
                }
                if (!isMax && !isMin) { break; }
            }
            
            if (isMax || isMin) {
                // Edge check
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
                    isFeature = true;
                }
            }
        }
    }

    // Aggregation
    var myWgIdx = 0u;
    if (isFeature) {
        myWgIdx = atomicAdd(&wgCount, 1u);
    }
    workgroupBarrier();

    if (lid == 0u) {
        let count = atomicLoad(&wgCount);
        if (count > 0u) {
            wgGlobalOffset = atomicAdd(&keypoints.count, count);
        }
    }
    workgroupBarrier();

    if (isFeature) {
        let idx = wgGlobalOffset + myWgIdx;
        // Write keypoint
        let scaleMult = pow(2.0, f32(params.octave));
        keypoints.points[idx].x = f32(x) * scaleMult;
        keypoints.points[idx].y = f32(y) * scaleMult;
        keypoints.points[idx].octave = f32(params.octave);
        keypoints.points[idx].scale = f32(params.scale);
        keypoints.points[idx].sigma = SIGMA_BASE * pow(2.0, (f32(params.scale) / SCALES_PER_OCTAVE)) * scaleMult;
        keypoints.points[idx].orientation = 0.0;
    }
}
