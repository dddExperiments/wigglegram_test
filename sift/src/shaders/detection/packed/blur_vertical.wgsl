struct Params {
    width: u32,  // Packed width
    height: u32, // Packed height
    radius: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTex: texture_2d<f32>; // Packed
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba32float, write>; // Packed
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

// Constants for shared memory optimization.
// These are fixed to match the default 16x16 workgroup size.
const MAX_RADIUS_PACKED: u32 = 16u;
const TILE_WIDTH_PACKED: u32 = 16u;
const TILE_HEIGHT_PACKED: u32 = 16u;
const CACHE_HEIGHT_PACKED: u32 = 48u; // TILE_HEIGHT_PACKED + 2 * MAX_RADIUS_PACKED

var<workgroup> tile_cache: array<vec4f, 768>; // CACHE_HEIGHT_PACKED * TILE_WIDTH_PACKED (48 * 16)

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u
) {
    let px = i32(gid.x);
    let py = i32(gid.y);
    let lx = lid.x;
    let ly = lid.y;
    let radius = i32(params.radius);
    
    // Shared memory layout: [ly + MAX_RADIUS_PACKED][lx]
    let center_idx = (ly + MAX_RADIUS_PACKED) * TILE_WIDTH_PACKED + lx;
    
    // Load packed texels (vec4f) into shared memory.
    
    // 1. Central packed texel
    if (px < i32(params.width) && py < i32(params.height)) {
        tile_cache[center_idx] = textureLoad(inputTex, vec2i(px, py), 0);
    } else {
        tile_cache[center_idx] = vec4f(0.0);
    }
    
    // 2. Top halo (16 packed texels up)
    let top_py = py - i32(TILE_HEIGHT_PACKED);
    tile_cache[center_idx - TILE_HEIGHT_PACKED * TILE_WIDTH_PACKED] = textureLoad(inputTex, vec2i(px, clamp(top_py, 0, i32(params.height) - 1)), 0);
    
    // 3. Bottom halo (16 packed texels down)
    let bot_py = py + i32(TILE_HEIGHT_PACKED);
    tile_cache[center_idx + TILE_HEIGHT_PACKED * TILE_WIDTH_PACKED] = textureLoad(inputTex, vec2i(px, clamp(bot_py, 0, i32(params.height) - 1)), 0);
    
    // Synchronize to ensure all threads have finished loading into tile_cache
    workgroupBarrier();
    
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    var sumCol0_0 = 0.0; // (2px, 2py)
    var sumCol0_1 = 0.0; // (2px, 2py+1)
    var sumCol1_0 = 0.0; // (2px+1, 2py)
    var sumCol1_1 = 0.0; // (2px+1, 2py+1)
    
    for (var k = -radius; k <= radius; k++) {
        let weight = kernel[u32(k + radius)];

        // Logical Y coords
        let ly0 = clamp(py * 2 + k, 0, i32(params.height) * 2 - 1);
        let ly1 = clamp(py * 2 + 1 + k, 0, i32(params.height) * 2 - 1);

        // Fetch values from shared memory cache
        // Packed relative offset: p_sy_rel = (ly0 / 2) - py
        let py0_rel = (ly0 / 2) - py;
        let py0_mod = ly0 % 2; // 0 (top/xy) or 1 (bot/zw)
        let v0 = tile_cache[i32(center_idx) + py0_rel * i32(TILE_WIDTH_PACKED)];
        
        let py1_rel = (ly1 / 2) - py;
        let py1_mod = ly1 % 2;
        let v1 = tile_cache[i32(center_idx) + py1_rel * i32(TILE_WIDTH_PACKED)];
        
        sumCol0_0 += select(v0.x, v0.z, py0_mod == 1) * weight;
        sumCol0_1 += select(v1.x, v1.z, py1_mod == 1) * weight;
        sumCol1_0 += select(v0.y, v0.w, py0_mod == 1) * weight;
        sumCol1_1 += select(v1.y, v1.w, py1_mod == 1) * weight;
    }
    
    // Output: (SumC0_0, SumC1_0, SumC0_1, SumC1_1)
    textureStore(outputTex, vec2i(px, py), vec4f(sumCol0_0, sumCol1_0, sumCol0_1, sumCol1_1));
}
