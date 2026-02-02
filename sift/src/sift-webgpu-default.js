import { SIFTWebGPUBase } from './sift-webgpu-base.js';

export class SIFTWebGPUDefault extends SIFTWebGPUBase {
    constructor(options) {
        super(options);
        this.inputTexture = null;
        this.buffers = {};
    }

    async init() {
        await super.init();
        await this.initPipelines();
        this.initBuffers();
    }

    async initPipelines() {
        const basePath = new URL('./shaders/detection/default', import.meta.url).href;
        await super.initPipelines(basePath);

        const commonPath = new URL('./shaders/common/prepare_dispatch.wgsl', import.meta.url).href;
        this.pipelines.prepareDispatch = this.createComputePipeline('prepareDispatch', await this.loadShader(commonPath));
    }

    initBuffers() {
        // Keypoint List Buffer: 16 bytes (count + 3 pads) + N * 32 bytes (Keypoint struct)
        // Shader has KeypointList { count, pad1, pad2, pad3, points[] }
        const kpSize = 16 + this.options.maxKeypoints * 32;
        this.buffers.keypoints = this.createBuffer(kpSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);

        // Descriptor Buffer: 128 floats * N (packed tightly? No, array<f32>)
        // WebGPU buffer alignment usually strict. 
        // 128 * 4 bytes = 512 bytes per descriptor.
        const descSize = this.options.maxKeypoints * 128 * 4;
        this.buffers.descriptors = this.createBuffer(descSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        // Params buffers (reused)
        this.buffers.params16 = this.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); // Generic 4-int params
        this.buffers.paramsExtrema = this.createBuffer(24, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); // Extrema params

        // Debug Histograms
        // 36 bins * 4 bytes * MAX_KEYPOINTS
        this.buffers.debugHist = this.createBuffer(this.options.maxKeypoints * 36 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        // Indirect dispatch buffer: 6 u32 values (orientation x,y,z + descriptor x,y,z)
        this.buffers.indirectDispatch = this.createBuffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);

        // Unified Params Buffer (Uniform) - for consolidated submissions
        this.buffers.unifiedParams = this.createBuffer(65536, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.paramsOffset = 0;
    }

    writeParams(data) {
        const offset = Math.ceil(this.paramsOffset / 256) * 256;
        if (offset + data.byteLength > 65536) {
            this.paramsOffset = 0;
            return this.writeParams(data);
        }
        this.device.queue.writeBuffer(this.buffers.unifiedParams, offset, data);
        this.paramsOffset = offset + data.byteLength;
        return offset;
    }

    resize(w, h) {
        if (this.pyramidCache && this.pyramidCache.w === w && this.pyramidCache.h === h) {
            return;
        }
        this.destroyPyramids();

        const gaussianPyramid = [], dogPyramid = [], octaveSizes = [];
        let currW = w, currH = h;

        // Base texture
        const baseTexture = this.createStorageTexture(currW, currH);

        for (let o = 0; o < this.options.numOctaves; o++) {
            octaveSizes.push({ w: currW, h: currH });
            const gaussOctave = [], dogOctave = [];

            for (let s = 0; s < this.options.scalesPerOctave + 3; s++) gaussOctave.push(this.createStorageTexture(currW, currH));
            for (let s = 0; s < this.options.scalesPerOctave + 2; s++) dogOctave.push(this.createStorageTexture(currW, currH));

            gaussianPyramid.push(gaussOctave);
            dogPyramid.push(dogOctave);

            currW = Math.floor(currW / 2); currH = Math.floor(currH / 2);
        }

        const tempTexture = this.createStorageTexture(w, h);

        this.pyramidCache = {
            w, h,
            baseTexture,
            gaussianPyramid,
            dogPyramid,
            octaveSizes,
            tempTexture
        };
        this.log(`[GPU-Default] Pyramids Allocated: ${w}x${h}`);
    }

    destroyPyramids() {
        if (!this.pyramidCache) return;
        this.pyramidCache.baseTexture.destroy();
        for (let oct of this.pyramidCache.gaussianPyramid) oct.forEach(t => t.destroy());
        for (let oct of this.pyramidCache.dogPyramid) oct.forEach(t => t.destroy());
        this.pyramidCache.tempTexture.destroy();
        this.pyramidCache = null;
    }

    async detectKeypoints(image) {
        const t0 = performance.now();
        this.timings = {};
        this.keypoints = [];

        const bitmap = await this.ensureImage(image);
        this.width = bitmap.width;
        this.height = bitmap.height;

        this.inputTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.inputTexture }, [this.width, this.height]);

        this.resize(this.width, this.height);
        const cache = this.pyramidCache;

        // Reset keypoint count
        this.device.queue.writeBuffer(this.buffers.keypoints, 0, new Uint32Array([0]));
        this.paramsOffset = 0;

        const enc = this.device.createCommandEncoder();

        // 1. Grayscale
        this.recordGrayscale(enc, cache.baseTexture);

        // 2. Pyramid
        this.recordPyramids(enc, cache);

        // 3. Extrema
        this.recordDetectExtrema(enc, cache.dogPyramid, cache.octaveSizes);

        // 3.5 Prepare indirect dispatch
        this.recordPrepareDispatch(enc);

        // 4. Orientation
        this.recordComputeOrientations(enc, cache.gaussianPyramid, cache.octaveSizes);

        this.device.queue.submit([enc.finish()]);

        // 5. Readback keypoints
        const t4 = performance.now();
        await this.readbackKeypoints();
        this.timings.readback = performance.now() - t4;

        if (this.inputTexture) this.inputTexture.destroy();

        this.log(`[GPU-Default] Detected ${this.keypoints.length} features`);
        return this.keypoints;
    }

    async computeDescriptors(keypoints) {
        const t0 = performance.now();
        if (!this.pyramidCache) throw new Error("Pyramids not allocated");

        await this.uploadKeypoints(keypoints);
        this.paramsOffset = 0;

        const enc = this.device.createCommandEncoder();
        this.recordPrepareDispatch(enc);

        const cache = this.pyramidCache;
        this.recordComputeDescriptors(enc, cache.gaussianPyramid, cache.octaveSizes);
        this.device.queue.submit([enc.finish()]);

        this.timings.descriptors = performance.now() - t0;

        const descriptors = await this.readbackDescriptors(keypoints.length);
        for (let i = 0; i < keypoints.length; i++) {
            keypoints[i].descriptor = descriptors[i];
        }
        return keypoints;
    }

    recordPrepareDispatch(enc) {
        const bind = this.device.createBindGroup({
            layout: this.pipelines.prepareDispatch.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.keypoints, size: 16 } },
                { binding: 1, resource: { buffer: this.buffers.indirectDispatch, size: 24 } }
            ]
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.prepareDispatch);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    async uploadKeypoints(keypoints) {
        // Default struct: Keypoint { x, y, octave, scale, sigma, orientation, padding... }
        // Size 32 bytes.
        // KeypointList: count, pad, pad, pad, keypoints[]
        const count = keypoints.length;
        const countData = new Uint32Array([count, 0, 0, 0]);
        this.device.queue.writeBuffer(this.buffers.keypoints, 0, countData);

        const kpData = new Float32Array(count * 8);
        for (let i = 0; i < count; i++) {
            const kp = keypoints[i];
            const off = i * 8;
            kpData[off + 0] = kp.x;
            kpData[off + 1] = kp.y;
            kpData[off + 2] = kp.octave;
            kpData[off + 3] = kp.scale;
            kpData[off + 4] = kp.sigma;
            kpData[off + 5] = kp.orientation;
            kpData[off + 6] = 0;
            kpData[off + 7] = 0;
        }
        this.device.queue.writeBuffer(this.buffers.keypoints, 16, kpData);
    }

    recordComputeDescriptors(enc, gaussianPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o];
            const octaveParams = new Int32Array([w, h, o, 0]);
            const offset = this.writeParams(octaveParams);

            const bind = this.device.createBindGroup({
                layout: this.pipelines.descriptor.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 16 } },
                    { binding: 1, resource: { buffer: this.buffers.keypoints } },
                    { binding: 2, resource: { buffer: this.buffers.descriptors } },
                    { binding: 3, resource: gaussianPyramid[o][1].createView() },
                    { binding: 4, resource: gaussianPyramid[o][2].createView() },
                    { binding: 5, resource: gaussianPyramid[o][3].createView() }
                ]
            });

            const pass = enc.beginComputePass();
            pass.setPipeline(this.pipelines.descriptor);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroupsIndirect(this.buffers.indirectDispatch, 12); // Offset 12: descriptor dispatch params
            pass.end();
        }
    }


    recordPyramids(enc, cache) {
        let w = this.width, h = this.height;
        const { gaussianPyramid, dogPyramid, octaveSizes } = cache;
        const tempTex = cache.tempTexture;

        for (let o = 0; o < this.options.numOctaves; o++) {
            const gaussOctave = gaussianPyramid[o];
            const dogOctave = dogPyramid[o];

            if (o === 0) {
                this.recordBlur(enc, cache.baseTexture, gaussOctave[0], tempTex, w, h, this.options.sigmaBase);
            } else {
                const prev = gaussianPyramid[o - 1][this.options.scalesPerOctave];
                const pw = octaveSizes[o - 1].w, ph = octaveSizes[o - 1].h;
                this.recordDownsample(enc, prev, gaussOctave[0], pw, ph, w, h);
            }

            for (let s = 1; s < gaussOctave.length; s++) {
                const sigma = this.getSigma(s);
                const prevSigma = this.getSigma(s - 1);
                const diff = Math.sqrt(sigma * sigma - prevSigma * prevSigma);
                this.recordBlur(enc, gaussOctave[s - 1], gaussOctave[s], tempTex, w, h, diff);
            }

            for (let s = 0; s < dogOctave.length; s++) {
                this.recordDoG(enc, gaussOctave[s], gaussOctave[s + 1], dogOctave[s], w, h);
            }

            w = Math.floor(w / 2); h = Math.floor(h / 2);
        }
    }

    async readbackKeypoints() {
        const countBuf = this.createBuffer(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.keypoints, 0, countBuf, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        await countBuf.mapAsync(GPUMapMode.READ);
        const count = new Uint32Array(countBuf.getMappedRange())[0];
        countBuf.unmap();
        countBuf.destroy();

        if (count === 0) return;

        const kpByteSize = count * 32;
        const readBuf = this.createBuffer(kpByteSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.keypoints, 16, readBuf, 0, kpByteSize);
        this.device.queue.submit([enc.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const kpData = new Float32Array(readBuf.getMappedRange());

        for (let i = 0; i < count; i++) {
            const off = i * 8;
            this.keypoints.push({
                x: kpData[off + 0] * this.scaleRestoreFactor,
                y: kpData[off + 1] * this.scaleRestoreFactor,
                octave: kpData[off + 2],
                scale: kpData[off + 3] * this.scaleRestoreFactor,
                sigma: kpData[off + 4],
                orientation: kpData[off + 5]
            });
        }
        readBuf.unmap(); readBuf.destroy();
    }

    async readbackDescriptors(count) {
        if (count === 0) return [];
        const descByteSize = count * 128 * 4;
        const descReadBuf = this.createBuffer(descByteSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.descriptors, 0, descReadBuf, 0, descByteSize);
        this.device.queue.submit([enc.finish()]);

        await descReadBuf.mapAsync(GPUMapMode.READ);
        const descData = new Float32Array(descReadBuf.getMappedRange());

        const descriptors = [];
        for (let i = 0; i < count; i++) {
            descriptors.push(new Float32Array(descData.slice(i * 128, (i + 1) * 128)));
        }
        descReadBuf.unmap(); descReadBuf.destroy();
        return descriptors;
    }

    recordGrayscale(enc, outputTex) {
        // Output dims are same as input for Default (Unpacked)
        const w = this.width;
        const h = this.height;

        const bind = this.device.createBindGroup({
            layout: this.pipelines.grayscale.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.inputTexture.createView() },
                { binding: 1, resource: outputTex.createView() }
            ]
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.grayscale);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        pass.end();
    }

    recordBlur(enc, inTex, outTex, tempTex, w, h, sigma) {
        const radius = Math.ceil(sigma * 3);
        const kernelBuf = this.getKernelBuffer(sigma, radius);

        const params = new Uint32Array([w, h, radius, 0]);
        const offset = this.writeParams(params);

        const bindH = this.device.createBindGroup({
            layout: this.pipelines.blurH.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 16 } },
                { binding: 1, resource: inTex.createView() },
                { binding: 2, resource: tempTex.createView() },
                { binding: 3, resource: { buffer: kernelBuf } }
            ]
        });

        const passH = enc.beginComputePass();
        passH.setPipeline(this.pipelines.blurH);
        passH.setBindGroup(0, bindH);
        passH.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        passH.end();

        const bindV = this.device.createBindGroup({
            layout: this.pipelines.blurV.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 16 } },
                { binding: 1, resource: tempTex.createView() },
                { binding: 2, resource: outTex.createView() },
                { binding: 3, resource: { buffer: kernelBuf } }
            ]
        });

        const passV = enc.beginComputePass();
        passV.setPipeline(this.pipelines.blurV);
        passV.setBindGroup(0, bindV);
        passV.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        passV.end();
    }

    recordDoG(enc, texA, texB, outTex, w, h) {
        const bind = this.device.createBindGroup({
            layout: this.pipelines.dog.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: texA.createView() },
                { binding: 1, resource: texB.createView() },
                { binding: 2, resource: outTex.createView() }
            ]
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.dog);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        pass.end();
    }

    recordDownsample(enc, inTex, outTex, sw, sh, dw, dh) {
        const params = new Uint32Array([sw, sh, dw, dh]);
        const offset = this.writeParams(params);

        const bind = this.device.createBindGroup({
            layout: this.pipelines.downsample.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 16 } },
                { binding: 1, resource: inTex.createView() },
                { binding: 2, resource: outTex.createView() }
            ]
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.downsample);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(dw / 16), Math.ceil(dh / 16));
        pass.end();
    }

    recordDetectExtrema(enc, dogPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o];
            for (let s = 1; s <= this.options.scalesPerOctave; s++) {
                const params = new ArrayBuffer(24);
                const view = new DataView(params);
                view.setInt32(0, w, true);
                view.setInt32(4, h, true);
                view.setInt32(8, o, true);
                view.setInt32(12, s, true);
                view.setFloat32(16, this.options.contrastThreshold / this.options.scalesPerOctave, true);
                view.setFloat32(20, this.options.edgeThreshold, true);
                const offset = this.writeParams(params);

                const bind = this.device.createBindGroup({
                    layout: this.pipelines.extrema.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 24 } },
                        { binding: 1, resource: dogPyramid[o][s - 1].createView() },
                        { binding: 2, resource: dogPyramid[o][s].createView() },
                        { binding: 3, resource: dogPyramid[o][s + 1].createView() },
                        { binding: 4, resource: { buffer: this.buffers.keypoints } }
                    ]
                });

                const pass = enc.beginComputePass();
                pass.setPipeline(this.pipelines.extrema);
                pass.setBindGroup(0, bind);
                pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
                pass.end();
            }
        }
    }

    recordComputeOrientations(enc, gaussianPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o];
            const octaveParams = new Int32Array([w, h, o]);
            const offset = this.writeParams(octaveParams);

            const bind = this.device.createBindGroup({
                layout: this.pipelines.orientation.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset, size: 12 } },
                    { binding: 1, resource: { buffer: this.buffers.keypoints } },
                    { binding: 2, resource: gaussianPyramid[o][1].createView() },
                    { binding: 3, resource: gaussianPyramid[o][2].createView() },
                    { binding: 4, resource: gaussianPyramid[o][3].createView() }
                ]
            });

            const pass = enc.beginComputePass();
            pass.setPipeline(this.pipelines.orientation);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroupsIndirect(this.buffers.indirectDispatch, 0); // Offset 0: orientation dispatch params
            pass.end();
        }
    }



    async readbackResults() {
        const countBuf = this.createBuffer(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.keypoints, 0, countBuf, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        await countBuf.mapAsync(GPUMapMode.READ);
        const count = new Uint32Array(countBuf.getMappedRange())[0];
        countBuf.unmap();
        countBuf.destroy();

        if (count === 0) return;

        const kpByteSize = count * 32;
        const descByteSize = count * 128 * 4;

        const readBuf = this.createBuffer(kpByteSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const descReadBuf = this.createBuffer(descByteSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

        const enc = this.device.createCommandEncoder();
        // Offset 16 bytes for padding in KeypointList
        enc.copyBufferToBuffer(this.buffers.keypoints, 16, readBuf, 0, kpByteSize);
        enc.copyBufferToBuffer(this.buffers.descriptors, 0, descReadBuf, 0, descByteSize);
        this.device.queue.submit([enc.finish()]);

        await Promise.all([
            readBuf.mapAsync(GPUMapMode.READ),
            descReadBuf.mapAsync(GPUMapMode.READ)
        ]);

        const kpData = new Float32Array(readBuf.getMappedRange());
        const descData = new Float32Array(descReadBuf.getMappedRange());

        for (let i = 0; i < count; i++) {
            const kp = {
                x: kpData[i * 8] * this.scaleRestoreFactor,
                y: kpData[i * 8 + 1] * this.scaleRestoreFactor,
                octave: kpData[i * 8 + 2],
                scale: kpData[i * 8 + 3] * this.scaleRestoreFactor,
                sigma: kpData[i * 8 + 4],
                orientation: kpData[i * 8 + 5],
                descriptor: new Float32Array(descData.slice(i * 128, (i + 1) * 128))
            };
            this.keypoints.push(kp);
        }

        readBuf.unmap(); readBuf.destroy();
        descReadBuf.unmap(); descReadBuf.destroy();
    }

}
