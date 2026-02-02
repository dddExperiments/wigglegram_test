#include "../../common/structs.wgsl"
#include "../../common/constants.wgsl"

struct Params {
    width: i32, height: i32, octave: i32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keypoints: KeypointList;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex2: texture_2d<f32>;
@group(0) @binding(4) var tex3: texture_2d<f32>;

var<workgroup> wgHist: array<atomic<u32>, ORI_BINS>;

fn getVal(s: i32, lx: i32, ly: i32) -> f32 {
    let px = lx / 2;
    let py = ly / 2;
    let mx = lx % 2;
    let my = ly % 2;
    
    var val: vec4f;
    if (s == 1) { val = textureLoad(tex1, vec2i(px, py), 0); }
    else if (s == 2) { val = textureLoad(tex2, vec2i(px, py), 0); }
    else { val = textureLoad(tex3, vec2i(px, py), 0); }
    
    if (my == 0) {
        return select(val.x, val.y, mx == 1);
    } else {
        return select(val.z, val.w, mx == 1);
    }
}

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE: u32 = 256u;

// 256 threads (1D) provides high occupancy and matches the thread count of 2D kernels (16x16).
// This is suitable for processing lists of keypoints.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let idx = wid.x + wid.y * 65535u;
    
    if (lid < ORI_BINS) {
        atomicStore(&wgHist[lid], 0u);
    }
    workgroupBarrier();

    let idxValid = idx < atomicLoad(&keypoints.count);
    var kp_octave = 0.0;
    var kp_x = 0.0;
    var kp_y = 0.0;
    var kp_scale = 0.0;
    
    if (idxValid) {
        let kp = keypoints.points[idx];
        kp_octave = kp.octave;
        kp_x = kp.x;
        kp_y = kp.y;
        kp_scale = kp.scale;
    }
    
    let isValid = idxValid && (i32(kp_octave) == params.octave);

    var x = 0;
    var y = 0;
    var scale = 0;
    var sigma = 0.0;
    var radius = 0;
    var radiusSq = 0.0;
    var width = 0;
    var totalPixels = 0;
    
    if (isValid) {
        x = i32(round(kp_x / pow(2.0, kp_octave)));
        y = i32(round(kp_y / pow(2.0, kp_octave)));
        scale = i32(kp_scale);
        
        sigma = SIGMA_BASE * pow(2.0, f32(scale) / SCALES_PER_OCTAVE); 
        radius = i32(round(sigma * 1.5 * 3.0));
        radiusSq = f32(radius * radius);
        width = 2 * radius + 1;
        totalPixels = width * width;
    }
    
    if (isValid && totalPixels > 0) {
        for (var i = i32(lid); i < totalPixels; i += 256) {
            let dy = (i / width) - radius;
            let dx = (i % width) - radius;
            
            let r2 = f32(dx*dx + dy*dy);
            if (r2 <= radiusSq) {
                let px = x + dx;
                let py = y + dy;
                // Check bounds (logical)
                if (px >= 1 && px < params.width * 2 - 1 && py >= 1 && py < params.height * 2 - 1) {
                    let rx = getVal(scale, px+1, py) - getVal(scale, px-1, py);
                    let ry = getVal(scale, px, py+1) - getVal(scale, px, py-1);
                    let mag = sqrt(rx*rx + ry*ry);
                    let sigma_w = 1.5 * sigma;
                    let weight = exp(-r2 / (2.0 * sigma_w * sigma_w));
                    
                    let ang_raw = atan2(ry, rx);
                    let ang = select(ang_raw, ang_raw + TWO_PI, ang_raw < 0.0);
                    let bin = i32(floor(ang * f32(ORI_BINS) / TWO_PI)) % i32(ORI_BINS);
                    
                    atomicAdd(&wgHist[bin], u32(mag * weight * HIST_SCALE));
                }
            }
        }
    }
    workgroupBarrier();

    if (lid == 0u && isValid) {
        // Read histogram
        var rawHist = array<f32, ORI_BINS>();
        for (var i = 0; i < i32(ORI_BINS); i++) {
            rawHist[i] = f32(atomicLoad(&wgHist[i])) / HIST_SCALE;
        }

        // Smooth Histogram [0.25, 0.5, 0.25]
        var histFloats = array<f32, ORI_BINS>();
        for (var i = 0; i < i32(ORI_BINS); i++) {
            let prev = rawHist[(i + i32(ORI_BINS) - 1) % i32(ORI_BINS)];
            let curr = rawHist[i];
            let next = rawHist[(i + 1) % i32(ORI_BINS)];
            histFloats[i] = 0.25 * prev + 0.5 * curr + 0.25 * next;
        }

        var maxVal = -1.0;
        var maxBin = 0;
        
        for (var i = 0; i < i32(ORI_BINS); i++) {
            let val = histFloats[i];
            if (val > maxVal) {
                maxVal = val;
                maxBin = i;
            }
        }
        
        let left = histFloats[(maxBin + i32(ORI_BINS) - 1) % i32(ORI_BINS)];
        let right = histFloats[(maxBin + 1) % i32(ORI_BINS)];
        let peak = f32(maxBin) + 0.5 * (left - right) / (left - 2.0 * maxVal + right);
        let orientation = peak * TWO_PI / f32(ORI_BINS);
        
        keypoints.points[idx].orientation = orientation;
    }
}
