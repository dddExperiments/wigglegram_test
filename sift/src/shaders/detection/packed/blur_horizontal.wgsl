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

// Unpack helper: Get logical pixel value at (lx, ly) 
// knowing that (px, py) contains [ (2px, 2py), (2px+1, 2py), (2px, 2py+1), (2px+1, 2py+1) ]
// But fetching randomly is slow. We should iterate intelligently.

// Constants for shared memory optimization.
// These are fixed to match the default 16x16 workgroup size.
const MAX_RADIUS_PACKED: u32 = 16u;
const TILE_WIDTH_PACKED: u32 = 16u;
const TILE_HEIGHT_PACKED: u32 = 16u;
const CACHE_WIDTH_PACKED: u32 = 48u; // TILE_WIDTH_PACKED + 2 * MAX_RADIUS_PACKED

var<workgroup> tile_cache: array<vec4f, 768>; // TILE_HEIGHT_PACKED * CACHE_WIDTH_PACKED (16 * 48)

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

    // Shared memory row baseline
    let row_offset = ly * CACHE_WIDTH_PACKED;
    let center_idx = row_offset + MAX_RADIUS_PACKED + lx;

    // Load packed texels (vec4f) into shared memory.
    // Each packed texel contains 4 logical pixels.
    
    // 1. Central packed texel
    if (px < i32(params.width) && py < i32(params.height)) {
        tile_cache[center_idx] = textureLoad(inputTex, vec2i(px, py), 0);
    } else {
        tile_cache[center_idx] = vec4f(0.0);
    }

    // 2. Left halo (16 packed texels left)
    let left_px = px - i32(TILE_WIDTH_PACKED);
    tile_cache[center_idx - TILE_WIDTH_PACKED] = textureLoad(inputTex, vec2i(clamp(left_px, 0, i32(params.width) - 1), py), 0);
    
    // 3. Right halo (16 packed texels right)
    let right_px = px + i32(TILE_WIDTH_PACKED);
    tile_cache[center_idx + TILE_WIDTH_PACKED] = textureLoad(inputTex, vec2i(clamp(right_px, 0, i32(params.width) - 1), py), 0);

    // Synchronize to ensure all threads have finished loading into tile_cache
    workgroupBarrier();

    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    var sumRow0_0 = 0.0; // For pixel (2px, 2py)
    var sumRow0_1 = 0.0; // For pixel (2px+1, 2py)
    var sumRow1_0 = 0.0; // For pixel (2px, 2py+1)
    var sumRow1_1 = 0.0; // For pixel (2px+1, 2py+1)
    
    for (var k = -radius; k <= radius; k++) {
        let weight = kernel[u32(k + radius)];
        
        // --- Row 0 ---
        // Target 0: sx = 2*px. Neighbor = 2*px + k.
        // Target 1: sx = 2*px+1. Neighbor = 2*px + 1 + k.
        let lx0 = clamp(px * 2 + k, 0, i32(params.width) * 2 - 1);
        let lx1 = clamp(px * 2 + 1 + k, 0, i32(params.width) * 2 - 1);
        
        // Fetch values from shared memory cache
        // Packed relative offset: p_sx_rel = (lx0 / 2) - px
        let p0_x_rel = (lx0 / 2) - px;
        let p0_mod = lx0 % 2;
        let val0_packed = tile_cache[i32(center_idx) + p0_x_rel];
        let val0 = select(val0_packed.x, val0_packed.y, p0_mod == 1);
        sumRow0_0 += val0 * weight;
        sumRow1_0 += select(val0_packed.z, val0_packed.w, p0_mod == 1) * weight;
        
        let p1_x_rel = (lx1 / 2) - px;
        let p1_mod = lx1 % 2;
        let val1_packed = tile_cache[i32(center_idx) + p1_x_rel];
        let val1 = select(val1_packed.x, val1_packed.y, p1_mod == 1);
        sumRow0_1 += val1 * weight;
        sumRow1_1 += select(val1_packed.z, val1_packed.w, p1_mod == 1) * weight;
    }
    
    // Store Result
    textureStore(outputTex, vec2i(px, py), vec4f(sumRow0_0, sumRow0_1, sumRow1_0, sumRow1_1));
}
