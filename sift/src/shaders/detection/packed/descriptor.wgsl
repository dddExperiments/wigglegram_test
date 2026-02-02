#include "../../common/structs.wgsl"
#include "../../common/constants.wgsl"
struct DescriptorList {
    data: array<f32>
}
struct Params {
    width: i32, height: i32, octave: i32, pad: i32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keypoints: KeypointList;
@group(0) @binding(2) var<storage, read_write> descriptors: DescriptorList;
@group(0) @binding(3) var tex1: texture_2d<f32>;
@group(0) @binding(4) var tex2: texture_2d<f32>;
@group(0) @binding(5) var tex3: texture_2d<f32>;

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


fn getValBilinear(s: i32, x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = x - f32(x0);
    let wy = y - f32(y0);
    
    // Bounds check handled by getVal implicitly (textureLoad safe? No, should be careful)
    // But descriptor loop checks bounds.
    
    let v00 = getVal(s, x0, y0);
    let v10 = getVal(s, x1, y0);
    let v01 = getVal(s, x0, y1);
    let v11 = getVal(s, x1, y1);
    
    return mix(mix(v00, v10, wx), mix(v01, v11, wx), wy);
}

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE: u32 = 64u;

// 64 threads is chosen for descriptor generation as it involves more registers per thread.
// This preserves high occupancy while allowing sufficient resources for trilinear interpolation.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= atomicLoad(&keypoints.count)) { return; }

    let kp = keypoints.points[idx];
    if (i32(kp.octave) != params.octave) { return; }

    let x = kp.x / pow(2.0, kp.octave);
    let y = kp.y / pow(2.0, kp.octave);
    let scale = i32(kp.scale);
    let angle = kp.orientation;
    let cos_t = cos(angle);
    let sin_t = sin(angle);
    
    // Scale-dependent step size
    let sigma = SIGMA_BASE * pow(2.0, f32(scale) / SCALES_PER_OCTAVE);
    let step = DESC_STEP_FACTOR * sigma; // 16 samples covers ~12 sigma

    var desc = array<f32, DESC_DIM>();
    for (var k=0u; k<DESC_DIM; k++) { desc[k] = 0.0; }
    

    for (var r = -8; r < 8; r++) {
        for (var c = -8; c < 8; c++) {
            // Revert to Original Rotation Logic but with Step Scaling
            // Fixed rotation application
            let rot_x = step * (f32(c)*cos_t - f32(r)*sin_t);
            let rot_y = step * (f32(c)*sin_t + f32(r)*cos_t);
            
            let sample_x = x + rot_x; // Keep as f32
            let sample_y = y + rot_y; // Keep as f32
            
            // Bounds check (ensure x+1/y+1 are valid)
            if (sample_x < 2.0 || sample_x >= f32(params.width * 2) - 2.0 || 
                sample_y < 2.0 || sample_y >= f32(params.height * 2) - 2.0) { continue; }
            
            let dx = getValBilinear(scale, sample_x+1.0, sample_y) - getValBilinear(scale, sample_x-1.0, sample_y);
            let dy = getValBilinear(scale, sample_x, sample_y+1.0) - getValBilinear(scale, sample_x, sample_y-1.0);
            
            let mag = sqrt(dx*dx + dy*dy);
            let ori = atan2(dy, dx) - angle;
            
            var n_ori = ori;
            while (n_ori < 0.0) { n_ori += TWO_PI; }
            while (n_ori >= TWO_PI) { n_ori -= TWO_PI; }
            
            // Trilinear Interpolation
            // -0.5 to center the 4x4 bins (range 0-4 covers -8 to 8 pixels)
            let rbin = (f32(r) + 8.0) / f32(DESC_SUBGRID_SIZE) - 0.5;
            let cbin = (f32(c) + 8.0) / f32(DESC_SUBGRID_SIZE) - 0.5;
            let obin = n_ori * f32(DESC_BINS) / TWO_PI;
            
            let mag_w = mag * exp(-(f32(r*r + c*c)) / DESC_GAUSSIAN_WEIGHT_SIGMA_SQ);
            
            for (var dr = 0; dr < 2; dr++) {
                let ri = i32(floor(rbin)) + dr;
                if (ri >= 0 && ri < 4) {
                    let r_w = select(1.0 - fract(rbin), fract(rbin), dr == 1);
                    
                    for (var dc = 0; dc < 2; dc++) {
                        let ci = i32(floor(cbin)) + dc;
                        if (ci >= 0 && ci < 4) {
                            let c_w = select(1.0 - fract(cbin), fract(cbin), dc == 1);
                            
                            for (var do_idx = 0; do_idx < 2; do_idx++) {
                                let oi_raw = i32(floor(obin)) + do_idx;
                                let o_w = select(1.0 - fract(obin), fract(obin), do_idx == 1);
                                
                                let oi = (oi_raw + i32(DESC_BINS)) % i32(DESC_BINS);
                                let idx = (ri * i32(DESC_SUBGRID_SIZE) + ci) * i32(DESC_BINS) + oi;
                                desc[idx] += mag_w * r_w * c_w * o_w;
                            }
                        }
                    }
                }
            }
        }
    }

    
    var norm = 0.0;
    for (var k=0u; k<DESC_DIM; k++) { norm += desc[k]*desc[k]; }
    norm = sqrt(norm) + 0.00001;
    
    for (var k=0u; k<DESC_DIM; k++) {
        desc[k] = min(0.2, desc[k] / norm);
    }
    
    norm = 0.0;
    for (var k=0u; k<DESC_DIM; k++) { norm += desc[k]*desc[k]; }
    norm = sqrt(norm) + 0.00001;
    
    for (var k=0u; k<DESC_DIM; k++) {
        descriptors.data[idx * DESC_DIM + k] = desc[k] / norm;
    }
}
