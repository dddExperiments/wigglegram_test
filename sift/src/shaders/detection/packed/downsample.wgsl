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
    
    // Destination is packed pixel (dx, dy) -> logical 2x2 block.
    // Logical coords: (2dx, 2dy), (2dx+1, 2dy), ...
    // Source coords should be 2x these logical coords?
    // Downsample is usually by picking every 2nd pixel (0, 2, 4...)
    // So Logical Src(sx, sy) = Logical Dst(dx, dy) * 2
    
    // Dst Pixel Components:
    // .x (TopLeft) -> Logical Dst(2dx, 2dy)     -> Src(4dx, 4dy)
    // .y (TopRight)-> Logical Dst(2dx+1, 2dy)   -> Src(4dx+2, 4dy)
    // .z (BotLeft) -> Logical Dst(2dx, 2dy+1)   -> Src(4dx, 4dy+2)
    // .w (BotRight)-> Logical Dst(2dx+1, 2dy+1) -> Src(4dx+2, 4dy+2)
    
    // Now map Logical Src to Packed Src:
    // P_Src(X, Y) contains Logical(2X..2X+1, 2Y..2Y+1)
    
    // 1. Src(4dx, 4dy):
    //    Packed X = 4dx / 2 = 2dx. Mod = 0.
    //    Packed Y = 4dy / 2 = 2dy. Mod = 0.
    //    Load Packed(2dx, 2dy). Component .x (TL)
    
    // 2. Src(4dx+2, 4dy):
    //    Packed X = (4dx+2)/2 = 2dx+1.
    //    Packed Y = 2dy.
    //    Load Packed(2dx+1, 2dy). Component .x (TL)
    
    // 3. Src(4dx, 4dy+2):
    //    Packed X = 2dx.
    //    Packed Y = 2dy+1.
    //    Load Packed(2dx, 2dy+1). Component .x (TL)
    
    // 4. Src(4dx+2, 4dy+2):
    //    Packed X = 2dx+1.
    //    Packed Y = 2dy+1.
    //    Load Packed(2dx+1, 2dy+1). Component .x (TL)
    
    // It seems we always sample the TL component (.x) of the packed source pixels!
    // Because we are downsampling by 2, and the packing is by 2.
    // So we skip every other PACKED pixel?
    // No, we skip every other LOGICAL pixel.
    // Logical indices: 0, 1, 2, 3, 4, 5...
    // Keep: 0, 2, 4...
    // Packed indices: 
    //   P0=[0,1], P1=[2,3], P2=[4,5]
    //   We want 0 (from P0.x), 2 (from P1.x), 4 (from P2.x).
    //   We skip P0.y (1), P0.z (row 1), P0.w
    //   Wait, row indices also skip.
    //   So yes, we only read .x components from specific packed pixels.
    
    let sx = gid.x * 2u;
    let sy = gid.y * 2u;
    
    let v0 = textureLoad(inputTex, vec2i(i32(sx), i32(sy)), 0).x;     // TL
    let v1 = textureLoad(inputTex, vec2i(i32(sx+1), i32(sy)), 0).x;   // TR
    let v2 = textureLoad(inputTex, vec2i(i32(sx), i32(sy+1)), 0).x;   // BL
    let v3 = textureLoad(inputTex, vec2i(i32(sx+1), i32(sy+1)), 0).x; // BR
    
    textureStore(outputTex, vec2i(gid.xy), vec4f(v0, v1, v2, v3));
}
