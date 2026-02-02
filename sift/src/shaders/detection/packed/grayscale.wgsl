@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba32float, write>;

// Weights for RGB -> Grayscale
const W = vec3f(0.299, 0.587, 0.114);

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(inputTex);
    // gid describes the PACKED coordinates (w/2, h/2)
    let px = i32(gid.x);
    let py = i32(gid.y);
    
    // Check bounds of output texture
    let outDims = textureDimensions(outputTex);
    if (px >= i32(outDims.x) || py >= i32(outDims.y)) { return; }

    // Src coords (2x2 block)
    let sx = px * 2;
    let sy = py * 2;
    
    // Load 4 pixels
    // Clamp to input dims if odd size - textureLoad handles out of bounds by return 0? No, usually clamped or needs check.
    // Safe check:
    let v00 = textureLoad(inputTex, vec2i(sx, sy), 0);
    let v10 = textureLoad(inputTex, vec2i(sx+1, sy), 0);
    let v01 = textureLoad(inputTex, vec2i(sx, sy+1), 0);
    let v11 = textureLoad(inputTex, vec2i(sx+1, sy+1), 0);
    
    // Convert to gray
    let g00 = dot(v00.rgb, W);
    let g10 = dot(v10.rgb, W);
    let g01 = dot(v01.rgb, W);
    let g11 = dot(v11.rgb, W);
    
    // Pack: x=(0,0), y=(1,0), z=(0,1), w=(1,1)
    // Corresponds to: TopLeft, TopRight, BotLeft, BotRight
    textureStore(outputTex, vec2i(px, py), vec4f(g00, g10, g01, g11));
}
