// Prepare indirect dispatch arguments based on keypoint count
// Reads the keypoint count and computes workgroup counts for descriptor/orientation shaders

struct KeypointHeader {
    count: atomic<u32>,
    pad1: u32,
    pad2: u32,
    pad3: u32
}

// Two sets of dispatch args: 
// - First 3 u32s: for orientation (1 keypoint per workgroup, 256 threads per WG)
// - Second 3 u32s: for descriptor (64 keypoints per workgroup)
struct DispatchArgs {
    // Orientation: uses wid.x + wid.y * 65535 indexing scheme
    ori_x: u32,
    ori_y: u32,
    ori_z: u32,
    // Descriptor: simple 1D dispatch, 64 keypoints per workgroup
    desc_x: u32,
    desc_y: u32,
    desc_z: u32
}

@group(0) @binding(0) var<storage, read_write> keypoints: KeypointHeader;
@group(0) @binding(1) var<storage, read_write> dispatch_args: DispatchArgs;

// Dispatching 1 thread as this kernel performs serial calculations for indirect dispatch arguments.
@compute @workgroup_size(1)
fn main() {
    let count = atomicLoad(&keypoints.count);
    
    // Orientation: 1 keypoint per workgroup, use 2D dispatch for large counts
    // Matches: let idx = wid.x + wid.y * 65535u in orientation.wgsl
    let ori_workgroups = max(count, 1u);
    if (ori_workgroups <= 65535u) {
        dispatch_args.ori_x = ori_workgroups;
        dispatch_args.ori_y = 1u;
    } else {
        dispatch_args.ori_x = 65535u;
        dispatch_args.ori_y = (ori_workgroups + 65534u) / 65535u;
    }
    dispatch_args.ori_z = 1u;
    
    // Descriptor: 64 keypoints per workgroup (workgroup_size(64))
    let desc_workgroups = (count + 63u) / 64u;
    dispatch_args.desc_x = max(desc_workgroups, 1u);
    dispatch_args.desc_y = 1u;
    dispatch_args.desc_z = 1u;
}

