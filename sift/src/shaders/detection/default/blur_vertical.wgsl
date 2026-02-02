struct Params {
    width: u32,
    height: u32,
    radius: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

// Constants for shared memory optimization.
// These are fixed to match the default 16x16 workgroup size.
const MAX_RADIUS: u32 = 16u;
const TILE_WIDTH: u32 = 16u;
const TILE_HEIGHT: u32 = 16u;
const CACHE_HEIGHT: u32 = 48u; // TILE_HEIGHT + 2 * MAX_RADIUS (16 + 32)

var<workgroup> tile_cache: array<f32, 768>; // CACHE_HEIGHT * TILE_WIDTH (48 * 16)

// Workgroup sizes can be specialized if needed for different GPU architectures.
// WARNING: If WG_SIZE_X or WG_SIZE_Y are changed via specialization constants,
// the shared memory tile_cache size must also be updated.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u
) {
    let radius = i32(params.radius);
    let lx = lid.x;
    let ly = lid.y;
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    
    // Shared memory layout: [ly + MAX_RADIUS][lx]
    let center_idx = (ly + MAX_RADIUS) * TILE_WIDTH + lx;
    
    // Load central pixels and halos into shared memory.
    // Each thread in the 16x16 block is responsible for loading 3 pixels vertically.
    
    // 1. Central pixel
    if (gx < i32(params.width) && gy < i32(params.height)) {
        tile_cache[center_idx] = textureLoad(inputTex, vec2i(gx, gy), 0).r;
    } else {
        tile_cache[center_idx] = 0.0;
    }
    
    // 2. Top halo (16 pixels up)
    let top_gy = gy - i32(TILE_HEIGHT);
    tile_cache[center_idx - TILE_HEIGHT * TILE_WIDTH] = textureLoad(inputTex, vec2i(gx, clamp(top_gy, 0, i32(params.height) - 1)), 0).r;
    
    // 3. Bottom halo (16 pixels down)
    let bot_gy = gy + i32(TILE_HEIGHT);
    tile_cache[center_idx + TILE_HEIGHT * TILE_WIDTH] = textureLoad(inputTex, vec2i(gx, clamp(bot_gy, 0, i32(params.height) - 1)), 0).r;
    
    // Synchronize to ensure all threads have finished loading into tile_cache
    workgroupBarrier();
    
    // Boundary check for computation
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    var sum: f32 = 0.0;
    for (var i: i32 = -radius; i <= radius; i++) {
        // Access shared memory instead of textureLoad
        sum += tile_cache[i32(center_idx) + i * i32(TILE_WIDTH)] * kernel[u32(i + radius)];
    }
    
    textureStore(outputTex, vec2i(gid.xy), vec4f(sum, 0.0, 0.0, 1.0));
}
