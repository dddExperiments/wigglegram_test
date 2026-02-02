# WebSIFTGPU API Documentation

## Overview

`WebSIFTGPU` is a high-performance implementation of the Scale-Invariant Feature Transform (SIFT) algorithm using WebGPU. It provides both a reference CPU implementation (JS) and two GPU implementations optimized for different use cases.

## Usage

```javascript
import { SIFTWebGPUPacked, MatcherWebGPU } from './src/index.js';

// 1. Initialize
const sift = new SIFTWebGPUPacked();
await sift.init();

// 2. Load Image (createImageBitmap is recommended)
const img = new Image();
img.src = 'path/to/image.jpg';
await img.decode();
const bitmap = await createImageBitmap(img);

// 3. Detect Features
const keypoints = await sift.detectAndCompute(bitmap);
// keypoints: Array of { x, y, scale, orientation, descriptor: Float32Array(128) }

// 4. Match (Optional)
const matcher = new MatcherWebGPU(sift.device);
await matcher.init();
const matches = await matcher.matchDescriptors(descA, descB);
```

---

## Classes

### `SIFTWebGPUDefault`

The standard implementation where each octave/scale is kept in separate textures. Good for debugging and understanding the pipeline.

**Methods:**
- `init()`: Async. Initializes the WebGPU device and pipelines.
- `detectKeypoints(image)`: Async. Takes an `ImageBitmap` or `HTMLImageElement`. Returns an array of keypoints *without* descriptors.
- `computeDescriptors(keypoints)`: Async. Computes descriptors for the given keypoints.
- `detectAndCompute(image)`: Async. Convenience method that runs detection and description.

### `SIFTWebGPUPacked` (Recommended)

An optimized implementation that packs 4 scales into RGBA texture channels. This significantly reduces texture lookups and improves performance.

**Methods:**
- Same interface as `SIFTWebGPUDefault`.

### `MatcherWebGPU`

A GPU-accelerated brute-force matcher using L2 distance (Euclidean).

**Methods:**
- `init()`: Async. Compiles the matching shader.
- `matchDescriptors(descriptorsA, descriptorsB, ratio = 0.75)`: Async.
    - `descriptorsA`, `descriptorsB`: Flattened `Float32Array` of descriptors.
    - `ratio`: Lowe's ratio test threshold (default 0.75).
    - Returns: Array of `[indexA, indexB]` pairs.

## Performance Tips

1.  **Reuse Instances**: Create the SIFT instance once and reuse it. `init()` is expensive.
2.  **Use `ImageBitmap`**: Pass `ImageBitmap` directly to avoiding main-thread decoding.
3.  **Packed Mode**: Always prefer `SIFTWebGPUPacked` for production.

