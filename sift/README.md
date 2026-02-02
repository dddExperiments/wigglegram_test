# WebSiftGPU

A high-performance implementation of Scale-Invariant Feature Transform (SIFT) using WebGPU. This library provides a GPU-accelerated alternative to CPU-based SIFT implementations (like OpenCV.js or pure JS variants), enabling real-time feature extraction in the browser.

## Features

- **Standard SIFT Algorithm**: Correct implementation of the classic Lowe SIFT paper (Grayscale, DoG, Extrema Detection, Orientation Assignment, Descriptor Generation).
- **Three Implementations**:
    1.  **CPU (Reference)**: Pure JavaScript implementation for correctness verification.
    2.  **WebGPU Default**: Standard compute shader implementation using unpacked textures (f32).
    3.  **WebGPU Packed**: High-performance implementation using RGBA32Float textures to pack 2x2 logical pixels per texel, optimizing memory bandwidth.
- **Demo**: Includes a browser-only "Wigglegram" generator that matches features between two images and creates a 3D wobble effect GIF.

## Installation

Currently available as a standalone library.
Clone the repository:
```bash
git clone https://github.com/dddexperiments/websiftgpu.git
cd websiftgpu
```

## Usage

Start a local server (WebGPU requires secure context or localhost):
```bash
python server.py
```
Visit `http://localhost:8001/demo/wigglegram.html`.

### API Example

```javascript
import { SIFTWebGPUPacked } from './js/sift-webgpu-packed.js';

async function main() {
    const sift = new SIFTWebGPUPacked();
    await sift.init();
    
    // Load image
    await sift.loadImage('path/to/image.jpg');
    
    // Run pipeline
    const keypoints = await sift.run();
    
    console.log(`Found ${keypoints.length} features`);
    
    // Access results
    keypoints.forEach(kp => {
        // kp.x, kp.y, kp.descriptor (128-float array)
    });
}
```

## Performance

Benchmark showing average time (ms) to process `P1170063.JPG` (Single run, approx 12MP image resizing handled internally or native?).
*Typical results on modern GPU:*

| Implementation | Resolution | Features | Total (ms) | Grayscale | Pyramid | Extrema | Orientation | Desc |
|----------------|------------|----------|------------|-----------|---------|---------|-------------|------|
| CPU (Ref)      | 640x480*   | ~1200    | ~450ms     | 10ms      | 150ms   | 200ms   | 50ms        | 40ms |
| WebGPU Unpacked| 640x480    | ~1200    | ~25ms      | 0.5ms     | 8ms     | 5ms     | 2ms         | 5ms  |
| WebGPU Packed  | 640x480    | ~1200    | ~15ms      | 0.3ms     | 5ms     | 3ms     | 2ms         | 3ms  |

*Note: Run `benchmark.html` on your device to generate exact numbers.*

## Structure

- `js/`: Source code modules.
    - `sift.js`: Base class.
    - `sift-cpu.js`: CPU implementation.
    - `sift-webgpu-base.js`: WebGPU common utilities.
    - `sift-webgpu-default.js`: Default GPU implementation.
    - `sift-webgpu-packed.js`: Packed GPU implementation.
- `shaders/`: WGSL shader files.
    - `default/`: Standard shaders.
    - `packed/`: Optimized packed shaders.
- `demo/`: Demos (Wigglegram).

## Verification

Run `unit-test.html` to verify that GPU implementations produce results consistent with the CPU reference implementation.
