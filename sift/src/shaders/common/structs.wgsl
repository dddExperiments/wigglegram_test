struct Keypoint {
    x: f32, y: f32, octave: f32, scale: f32, sigma: f32, orientation: f32, p1: f32, p2: f32
}
struct KeypointList {
    count: atomic<u32>, pad1: u32, pad2: u32, pad3: u32, points: array<Keypoint>
}
struct MatchResult {
    bestIdx: i32,
    bestDistSq: f32,
    secondDistSq: f32,
    pad: f32
}
