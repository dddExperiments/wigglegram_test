struct Params {
    srcWidth: u32,
    srcHeight: u32,
    dstWidth: u32,
    dstHeight: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba32float, write>;

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.dstWidth || gid.y >= params.dstHeight) { return; }
    
    let srcX = i32(gid.x * 2u);
    let srcY = i32(gid.y * 2u);
    
    let val = textureLoad(inputTex, vec2i(srcX, srcY), 0).r;
    textureStore(outputTex, vec2i(gid.xy), vec4f(val, 0.0, 0.0, 1.0));
}
