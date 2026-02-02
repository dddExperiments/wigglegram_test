import { Matcher } from './matcher.js';
import { processShaderIncludes } from './sift-webgpu-base.js';

export class MatcherWebGPU extends Matcher {
    constructor(device, options = {}) {
        super(options);
        this.device = device;
        this.matcherPipeline = null;
    }

    async init() {
        if (!this.device) throw new Error("No device provided");

        const shaderCode = await this.loadShader(new URL('./shaders/matching/matcher.wgsl', import.meta.url).href);
        const quantizedCode = await this.loadShader(new URL('./shaders/matching/matcher_quantized.wgsl', import.meta.url).href);
        const guidedCode = await this.loadShader(new URL('./shaders/matching/matcher_guided.wgsl', import.meta.url).href);

        this.matcherPipeline = this.device.createComputePipeline({
            label: 'matcher-float',
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: shaderCode }),
                entryPoint: 'main'
            }
        });

        this.matcherQuantizedPipeline = this.device.createComputePipeline({
            label: 'matcher-uint8',
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: quantizedCode }),
                entryPoint: 'main'
            }
        });

        this.matcherGuidedPipeline = this.device.createComputePipeline({
            label: 'matcher-guided',
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: guidedCode }),
                entryPoint: 'main'
            }
        });
    }

    async loadShader(path) {
        const res = await fetch(path);
        if (!res.ok) throw new Error(`Failed to load shader: ${path}`);
        let code = await res.text();
        return await processShaderIncludes(path, code);
    }

    /**
     * Run feature matching on GPU
     * @param {Float32Array} descriptorsA - Flat array of descriptors (countA * 128)
     * @param {Float32Array} descriptorsB - Flat array of descriptors (countB * 128)
     * @param {number} ratio - Ratio threshold
     * @returns {Promise<Array>} List of matches [[idxA, idxB], ...]
     */
    async match(descriptorsA, descriptorsB, ratio = 0.75) {
        if (!descriptorsA || !descriptorsB || descriptorsA.length === 0 || descriptorsB.length === 0) {
            return [];
        }

        const countA = descriptorsA.length / 128;
        const countB = descriptorsB.length / 128;

        // Create buffers
        // NOTE: alignment needs! mappedAtCreation handles this typically?
        // Float32Array is 4 bytes aligned. 
        // 128 floats * 4 bytes = 512 bytes. Aligned.

        const bufferA = this.device.createBuffer({
            size: descriptorsA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new descriptorsA.constructor(bufferA.getMappedRange()).set(descriptorsA);
        bufferA.unmap();

        const bufferB = this.device.createBuffer({
            size: descriptorsB.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new descriptorsB.constructor(bufferB.getMappedRange()).set(descriptorsB);
        bufferB.unmap();

        // Output: Struct MatchResult { bestIdx: i32, bestDist: f32, secondDist: f32, pad: f32 } = 16 bytes
        const resultSize = countA * 16;
        const resultBuffer = this.device.createBuffer({
            size: resultSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Params
        const paramsData = new Uint32Array([countA, countB, 0, 0]);
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const isQuantized = descriptorsA instanceof Uint8Array;
        const pipeline = isQuantized ? this.matcherQuantizedPipeline : this.matcherPipeline;

        // Bind Group
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: bufferA } },
                { binding: 2, resource: { buffer: bufferB } },
                { binding: 3, resource: { buffer: resultBuffer } }
            ]
        });

        // Run
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(countA / 64));
        pass.end();

        // Readback
        const readBuffer = this.device.createBuffer({
            size: resultSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultSize);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBuffer.getMappedRange();
        const results = new Float32Array(arrayBuffer);
        // results is floats, but bestIdx is int. We can access via Int32Array view
        const resultsInt = new Int32Array(arrayBuffer);

        const matches = [];
        const ratioSq = ratio * ratio; // Lowe's ratio test squared (since we use distSq)

        for (let i = 0; i < countA; i++) {
            const bestIdx = resultsInt[i * 4];
            const bestDistSq = results[i * 4 + 1];
            const secondDistSq = results[i * 4 + 2];

            if (bestIdx >= 0 && bestDistSq < ratioSq * secondDistSq) {
                matches.push([i, bestIdx]);
            }
        }

        readBuffer.unmap();

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        resultBuffer.destroy();
        readBuffer.destroy();
        paramsBuffer.destroy();

        return matches;
    }

    // Alias for backward compatibility
    async matchDescriptors(descriptorsA, descriptorsB, ratio = 0.75) {
        return this.match(descriptorsA, descriptorsB, ratio);
    }

    /**
     * Run guided matching on GPU using Fundamental Matrix constraint
     * @param {Float32Array} descriptorsA 
     * @param {Float32Array} descriptorsB 
     * @param {Float32Array} keypointsA - Flat array of [x, y, ...] (countA * 8 if packed, or just [x,y])
     * @param {Float32Array} keypointsB 
     * @param {Array<Array<number>>} F - 3x3 Fundamental Matrix
     * @param {number} threshold - Epipolar distance threshold in pixels
     * @returns {Promise<Array>} Matches
     */
    async matchGuided(descriptorsA, descriptorsB, keypointsA, keypointsB, F, threshold = 5.0) {
        if (!descriptorsA || !descriptorsB || descriptorsA.length === 0 || descriptorsB.length === 0) return [];

        const countA = descriptorsA.length / 128;
        const countB = descriptorsB.length / 128;

        // 1. Create Descriptor Buffers
        const bufferA = this.device.createBuffer({ size: descriptorsA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        new Float32Array(bufferA.getMappedRange()).set(descriptorsA);
        bufferA.unmap();

        const bufferB = this.device.createBuffer({ size: descriptorsB.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        new Float32Array(bufferB.getMappedRange()).set(descriptorsB);
        bufferB.unmap();

        // 2. Create Keypoint Buffers (only x,y needed)
        const kpABuffer = this.device.createBuffer({ size: countA * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        const kpAData = new Float32Array(kpABuffer.getMappedRange());
        for (let i = 0; i < countA; i++) {
            kpAData[i * 2] = keypointsA[i].x;
            kpAData[i * 2 + 1] = keypointsA[i].y;
        }
        kpABuffer.unmap();

        const kpBBuffer = this.device.createBuffer({ size: countB * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        const kpBData = new Float32Array(kpBBuffer.getMappedRange());
        for (let i = 0; i < countB; i++) {
            kpBData[i * 2] = keypointsB[i].x;
            kpBData[i * 2 + 1] = keypointsB[i].y;
        }
        kpBBuffer.unmap();

        // 3. Result Buffer
        const resultSize = countA * 16;
        const resultBuffer = this.device.createBuffer({ size: resultSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        // 4. Params (F matrix column-major, plus other params)
        // struct Params { countA: u32, countB: u32, threshold: f32, pad: u32, col0: vec4f, col1: vec4f, col2: vec4f }
        const paramsData = new Float32Array(4 + 12);
        const paramsU32 = new Uint32Array(paramsData.buffer);
        paramsU32[0] = countA;
        paramsU32[1] = countB;
        paramsData[2] = threshold;
        // F is 3x3. WGSL Params expects col0, col1, col2 as vec4f
        for (let col = 0; col < 3; col++) {
            for (let row = 0; row < 3; row++) {
                paramsData[4 + col * 4 + row] = F[row][col];
            }
        }

        const paramsBuffer = this.device.createBuffer({ size: paramsData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        // 5. Bind Group
        const bindGroup = this.device.createBindGroup({
            layout: this.matcherGuidedPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: bufferA } },
                { binding: 2, resource: { buffer: bufferB } },
                { binding: 3, resource: { buffer: resultBuffer } },
                { binding: 4, resource: { buffer: kpABuffer } },
                { binding: 5, resource: { buffer: kpBBuffer } }
            ]
        });

        // 6. Run
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.matcherGuidedPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(countA / 64));
        pass.end();

        // 7. Readback
        const readBuffer = this.device.createBuffer({ size: resultSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultSize);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(readBuffer.getMappedRange());
        const resultsInt = new Int32Array(results.buffer);

        const matches = [];
        for (let i = 0; i < countA; i++) {
            const bestIdx = resultsInt[i * 4];
            if (bestIdx >= 0) matches.push([i, bestIdx]);
        }
        readBuffer.unmap();

        // Cleanup
        [bufferA, bufferB, kpABuffer, kpBBuffer, resultBuffer, readBuffer, paramsBuffer].forEach(b => b.destroy());

        return matches;
    }
}
