#include "../../common/structs.wgsl"
#include "../../common/constants.wgsl"

struct DescriptorListQuantized {
    data: array<u32> // Packed 4x uint8 per u32
}
struct Params {
    width: i32, height: i32, octave: i32, pad: i32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keypoints: KeypointList;
@group(0) @binding(2) var<storage, read_write> descriptors: DescriptorListQuantized;
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
    let wx = x - f32(x0);
    let wy = y - f32(y0);
    
    let v00 = getVal(s, x0, y0);
    let v10 = getVal(s, x0 + 1, y0);
    let v01 = getVal(s, x0, y0 + 1);
    let v11 = getVal(s, x0 + 1, y0 + 1);
    
    return mix(mix(v00, v10, wx), mix(v01, v11, wx), wy);
}

@compute @workgroup_size(64)
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
    
    let sigma = SIGMA_BASE * pow(2.0, f32(scale)/SCALES_PER_OCTAVE);
    let step = DESC_STEP_FACTOR * sigma;

    var desc = array<f32, DESC_DIM>();
    for (var k=0u; k<DESC_DIM; k++) { desc[k] = 0.0; }

    for (var r = -8; r < 8; r++) {
        for (var c = -8; c < 8; c++) {
            let rot_x = step * (f32(c)*cos_t - f32(r)*sin_t);
            let rot_y = step * (f32(c)*sin_t + f32(r)*cos_t);
            let sample_x = x + rot_x;
            let sample_y = y + rot_y;
            
            if (sample_x < 2.0 || sample_x >= f32(params.width * 2) - 2.0 || 
                sample_y < 2.0 || sample_y >= f32(params.height * 2) - 2.0) { continue; }
            
            let dx = getValBilinear(scale, sample_x+1.0, sample_y) - getValBilinear(scale, sample_x-1.0, sample_y);
            let dy = getValBilinear(scale, sample_x, sample_y+1.0) - getValBilinear(scale, sample_x, sample_y-1.0);
            
            let mag = sqrt(dx*dx + dy*dy);
            var ori = atan2(dy, dx) - angle;
            while (ori < 0.0) { ori += TWO_PI; }
            while (ori >= TWO_PI) { ori -= TWO_PI; }
            
            let rbin = (f32(r) + 8.0) / f32(DESC_SUBGRID_SIZE) - 0.5;
            let cbin = (f32(c) + 8.0) / f32(DESC_SUBGRID_SIZE) - 0.5;
            let obin = ori * f32(DESC_BINS) / TWO_PI;
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
                                let d_idx = (ri * i32(DESC_SUBGRID_SIZE) + ci) * i32(DESC_BINS) + oi;
                                desc[d_idx] += mag_w * r_w * c_w * o_w;
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
    for (var k=0u; k<DESC_DIM; k++) { desc[k] = min(0.2, desc[k] / norm); }
    norm = 0.0;
    for (var k=0u; k<DESC_DIM; k++) { norm += desc[k]*desc[k]; }
    norm = sqrt(norm) + 0.00001;
    
    for (var k=0u; k<32u; k++) {
        let v1 = u32(clamp(desc[k*4u+0u] / norm * 512.0, 0.0, 255.0));
        let v2 = u32(clamp(desc[k*4u+1u] / norm * 512.0, 0.0, 255.0));
        let v3 = u32(clamp(desc[k*4u+2u] / norm * 512.0, 0.0, 255.0));
        let v4 = u32(clamp(desc[k*4u+3u] / norm * 512.0, 0.0, 255.0));
        descriptors.data[idx * 32u + k] = v1 | (v2 << 8u) | (v3 << 16u) | (v4 << 24u);
    }
}
