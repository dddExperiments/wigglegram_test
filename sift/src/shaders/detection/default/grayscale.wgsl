@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba32float, write>;

// Workgroup sizes can be specialized if needed for different GPU architectures.
override WG_SIZE_X: u32 = 16u;
override WG_SIZE_Y: u32 = 16u;

// 16x16 workgroup size (256 threads) is a balanced choice for 2D image processing
// ensuring high occupancy and efficient texture access across most GPU architectures.
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(inputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let color = textureLoad(inputTex, vec2i(gid.xy), 0);
    let gray = dot(color.rgb, vec3f(0.299, 0.587, 0.114));
    
    textureStore(outputTex, vec2i(gid.xy), vec4f(gray, 0.0, 0.0, 1.0));
}
