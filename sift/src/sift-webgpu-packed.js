import { SIFTWebGPUBase } from './sift-webgpu-base.js';

// import { NUM_OCTAVES, SCALES_PER_OCTAVE, SIGMA_BASE, CONTRAST_THRESHOLD, EDGE_THRESHOLD, MAX_KEYPOINTS } from './constants.js';

export class SIFTWebGPUPacked extends SIFTWebGPUBase {
    constructor(options = {}) {
        super(options);
        this.inputTexture = null;
        this.inputTextures = [null, null];
        this.currentInputIdx = 0;
        this.buffers = {};
        this.pyramidCache = null;
        this.pipelined = !!options.pipelined;
    }

    async init() {
        await super.init();
        await this.initPipelines();
        this.initBuffers();
    }

    async initPipelines() {
        const basePath = new URL('./shaders/detection/packed', import.meta.url).href;
        await super.initPipelines(basePath);

        // Load prepare_dispatch pipeline for indirect dispatch
        const commonPath = new URL('./shaders/common/prepare_dispatch.wgsl', import.meta.url).href;
        this.pipelines.prepareDispatch = this.createComputePipeline('prepareDispatch', await this.loadShader(commonPath));

        if (this.options.quantizeDescriptors) {
            this.pipelines.descriptor = this.createComputePipeline('descriptor_quantized', await this.loadShader(`${basePath}/descriptor_quantized.wgsl`));
        }
    }

    initBuffers() {
        const kpSize = 16 + this.options.maxKeypoints * 32;
        this.buffers.keypoints = this.createBuffer(kpSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);

        const descBytesPerFeature = this.options.quantizeDescriptors ? 128 : 512;
        const descSize = this.options.maxKeypoints * descBytesPerFeature;

        this.buffers.descriptors = this.createBuffer(descSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.buffers.params16 = this.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.buffers.paramsExtrema = this.createBuffer(24, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        // Indirect dispatch buffer: 6 u32 values (orientation x,y,z + descriptor x,y,z)
        this.buffers.indirectDispatch = this.createBuffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);

        this.numStages = 3; // Triple buffering
        this.buffers.staging = new Array(this.numStages).fill(0).map(() =>
            this.createBuffer(kpSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
        this.buffers.stagingDescriptors = new Array(this.numStages).fill(0).map(() =>
            this.createBuffer(descSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

        this.buffers.unifiedParams = this.createBuffer(65536, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.frameIdx = 0;
        this.mapPromises = new Array(this.numStages).fill(null);
        this.mapDescPromises = new Array(this.numStages).fill(null);
    }

    writeParams(enc, data) {
        // Align offset to 256 bytes
        const offset = Math.ceil(this.paramsOffset / 256) * 256;
        const size = data.byteLength;
        if (offset + size > this.buffers.unifiedParams.size) {
            console.warn("Unified Params Buffer Overflow! Resetting (may cause glitches if mid-frame)");
            this.paramsOffset = 0;
            return this.writeParams(enc, data);
        }

        // We can't use queue.writeBuffer inside an encoder session strictly?
        // Actually queue.writeBuffer calls are sequenced before queue.submit(). 
        // But we are recording commands. 
        // IMPORTANT: multiple writeBuffers to the same buffer in one frame might be batched?
        // WebGPU spec says queue writes happen before subsequent submits.
        // We are NOT submitting yet. 
        // Wait, if we use queue.writeBuffer MULTIPLE times to different offsets, it works fine.
        this.device.queue.writeBuffer(this.buffers.unifiedParams, offset, data);
        this.paramsOffset = offset + size;
        return offset;
    }

    resize(w, h) {
        if (this.pyramidCache && this.pyramidCache.w === w && this.pyramidCache.h === h) {
            return; // Already allocated
        }
        this.destroyPyramids();

        const gaussianPyramid = [], dogPyramid = [], octaveSizes = [];
        let pw = Math.ceil(w / 2), ph = Math.ceil(h / 2); // Packed Size

        // Base texture (packed)
        const baseTexture = this.createStorageTexture(pw, ph);

        let currW = pw, currH = ph;

        for (let o = 0; o < this.options.numOctaves; o++) {
            octaveSizes.push({ w: currW, h: currH });
            const gaussOctave = [], dogOctave = [];
            for (let s = 0; s < this.options.scalesPerOctave + 3; s++) gaussOctave.push(this.createStorageTexture(currW, currH));
            for (let s = 0; s < this.options.scalesPerOctave + 2; s++) dogOctave.push(this.createStorageTexture(currW, currH));

            gaussianPyramid.push(gaussOctave);
            dogPyramid.push(dogOctave);

            currW = Math.floor(currW / 2); currH = Math.floor(currH / 2);
        }

        const tempTexture = this.createStorageTexture(pw, ph);

        this.pyramidCache = {
            w, h,
            baseTexture,
            gaussianPyramid,
            dogPyramid,
            octaveSizes,
            tempTexture
        };

        this.log(`[GPU-Packed] Pyramids Allocated: ${w}x${h}`);
    }

    destroyPyramids() {
        if (!this.pyramidCache) return;
        this.pyramidCache.baseTexture.destroy();
        for (let oct of this.pyramidCache.gaussianPyramid) oct.forEach(t => t.destroy());
        for (let oct of this.pyramidCache.dogPyramid) oct.forEach(t => t.destroy());
        this.pyramidCache.tempTexture.destroy();
        this.pyramidCache = null;
    }

    /**
     * Pre-allocates resources for real-time video processing to avoid 
     * allocations during the main loop.
     * @param {number} width 
     * @param {number} height 
     */
    initForVideo(width, height) {
        this.width = width;
        this.height = height;

        // Allocate 2 input textures for ping-pong
        for (let i = 0; i < 2; i++) {
            if (this.inputTextures[i]) this.inputTextures[i].destroy();
            this.inputTextures[i] = this.device.createTexture({
                size: [width, height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
            });
        }
        this.inputTexture = this.inputTextures[0];

        // Pre-allocate pyramids
        this.resize(width, height);
    }

    /**
     * Processes a single video frame using the double-buffer system.
     * @param {HTMLVideoElement|ImageBitmap} source 
     * @returns {Promise<Array>}
     */
    async processVideoFrame(source) {
        // Ping-pong input texture
        this.currentInputIdx = (this.currentInputIdx + 1) % 2;
        this.inputTexture = this.inputTextures[this.currentInputIdx];

        if (!this.inputTexture) {
            // Fallback if not initialized for video
            this.inputTexture = await this.loadImage(source);
        } else {
            this.device.queue.copyExternalImageToTexture(
                { source: source },
                { texture: this.inputTexture },
                [this.width, this.height]
            );
        }

        // Run detection (pipelined if enabled)
        if (this.pipelined) {
            return this.detectAndComputeWithPipelining();
        } else {
            return this.detectAndComputeOnCurrent();
        }
    }

    /**
     * Runs the pipeline on the currently uploaded inputTexture.
     * Assumes this.inputTexture and this.pyramidCache are ready.
     */
    async detectAndComputeOnCurrent() {
        this.timings = {};
        this.keypoints = [];
        this.paramsOffset = 0;
        const cache = this.pyramidCache;

        this.device.queue.writeBuffer(this.buffers.keypoints, 0, new Uint32Array([0]));
        const enc = this.device.createCommandEncoder();
        this.recordDetection(enc, cache);
        this.recordComputeDescriptors(enc, cache.gaussianPyramid, cache.octaveSizes);
        this.recordReadbackKeypoints(enc);
        this.recordReadbackDescriptors(enc);
        this.device.queue.submit([enc.finish()]);

        const t_read = performance.now();
        await this.awaitReadbackKeypoints();
        const descriptors = await this.awaitReadbackDescriptors(true);
        this.timings.readback = performance.now() - t_read;

        for (let i = 0; i < this.keypoints.length; i++) {
            this.keypoints[i].descriptor = descriptors[i];
        }
        return this.keypoints;
    }

    /**
     * Runs the pipeline in a pipelined fashion. 
     * Returns results for the PREVIOUS frame while the CURRENT frame is on the GPU.
     */
    async detectAndComputeWithPipelining() {
        this.paramsOffset = 0;
        const cache = this.pyramidCache;

        this.device.queue.writeBuffer(this.buffers.keypoints, 0, new Uint32Array([0]));
        const enc = this.device.createCommandEncoder();
        this.recordDetection(enc, cache);
        this.recordComputeDescriptors(enc, cache.gaussianPyramid, cache.octaveSizes);
        this.recordReadbackKeypoints(enc);
        this.recordReadbackDescriptors(enc);
        this.device.queue.submit([enc.finish()]);

        // Trigger map for CURRENT frame (N)
        const kpMapPromise = this.triggerKeypointReadback();
        const descMapPromise = this.triggerDescriptorReadback();

        // Process results from PREVIOUS frame (N-1 or N-2 depending on pipeline depth)
        // With triple buffering, we can actually look back 1 frame safely.
        const [keypoints, descriptors] = await Promise.all([
            this.retrieveKeypoints(true),
            this.retrieveDescriptors(true, true) // returnFlat = true for speed
        ]);

        this.frameIdx++;

        if (keypoints && keypoints.length > 0) {
            // If returnFlat was true, descriptors is a single Float32Array/Uint8Array
            const isQuant = this.options.quantizeDescriptors;
            const step = 128;
            for (let i = 0; i < keypoints.length; i++) {
                if (isQuant) {
                    keypoints[i].descriptor = descriptors.slice(i * step, (i + 1) * step);
                } else {
                    keypoints[i].descriptor = descriptors.slice(i * step, (i + 1) * step);
                }
            }
        }

        this.keypoints = keypoints || [];
        return this.keypoints;
    }


    async detectKeypoints(image) {
        const t0 = performance.now();
        this.timings = {};
        this.keypoints = [];
        this.paramsOffset = 0; // Reset param buffer allocator

        // 1. Process Input
        const t_process = performance.now();
        const bitmap = await this.processBitmap(image);
        this.timings.process = performance.now() - t_process;

        const t_upload = performance.now();
        this.uploadImageToTexture(bitmap);
        this.timings.upload = performance.now() - t_upload;

        // 2. Resize / Allocate
        this.resize(this.width, this.height);
        const cache = this.pyramidCache;

        // Reset keypoint count
        this.device.queue.writeBuffer(this.buffers.keypoints, 0, new Uint32Array([0]));

        // --- ENCODING ---
        const enc = this.device.createCommandEncoder();

        this.recordDetection(enc, cache);
        this.recordReadbackKeypoints(enc);

        // --- SUBMIT ---
        this.device.queue.submit([enc.finish()]);

        // 8. Readback (Map & Parse)
        this.timings.grayscale = 0;
        this.timings.pyramid = 0;
        this.timings.extrema = 0;
        this.timings.orientation = 0;

        const t4 = performance.now();
        await this.awaitReadbackKeypoints();
        this.timings.readback = performance.now() - t4;

        this.log(`[GPU-Packed] Detected ${this.keypoints.length} features`);
        return this.keypoints;
    }

    async detectAndCompute(image) {
        this.timings = {};
        this.keypoints = [];
        this.paramsOffset = 0;

        // 1. Process Input
        const t_process = performance.now();
        const bitmap = await this.processBitmap(image);
        this.timings.process = performance.now() - t_process;

        const t_upload = performance.now();
        this.uploadImageToTexture(bitmap);
        this.timings.upload = performance.now() - t_upload;

        // 2. Resize / Allocate
        this.resize(this.width, this.height);
        const cache = this.pyramidCache;

        // Reset keypoint count
        this.device.queue.writeBuffer(this.buffers.keypoints, 0, new Uint32Array([0]));

        // --- ENCODING ---
        const enc = this.device.createCommandEncoder();

        // Detect
        this.recordDetection(enc, cache);

        // Compute Descriptors (Directly using the keypoints buffer)
        this.recordComputeDescriptors(enc, cache.gaussianPyramid, cache.octaveSizes);

        // Readback Both
        this.recordReadbackKeypoints(enc);
        this.recordReadbackDescriptors(enc);

        // --- SUBMIT ---
        this.device.queue.submit([enc.finish()]);

        // --- READBACK ---
        const t4 = performance.now();
        await this.awaitReadbackKeypoints(); // Populates this.keypoints
        const descriptors = await this.awaitReadbackDescriptors(true); // Returns Float32Array[]. True because keypoints incremented frameIdx
        this.timings.readback = performance.now() - t4;

        // Merge
        for (let i = 0; i < this.keypoints.length; i++) {
            this.keypoints[i].descriptor = descriptors[i];
        }

        return this.keypoints;
    }

    recordDetection(enc, cache) {
        const packedW = Math.ceil(this.width / 2);
        const packedH = Math.ceil(this.height / 2);

        // 3. Grayscale
        this.runGrayscale(enc, cache.baseTexture, packedW, packedH);

        // 4. Pyramids
        this.buildPyramids(enc, packedW, packedH, cache);

        // 5. Extrema
        this.detectExtrema(enc, cache.dogPyramid, cache.octaveSizes);

        // 6. Prepare indirect dispatch (compute workgroup counts based on actual keypoint count)
        this.runPrepareDispatch(enc);

        // 7. Orientation (using indirect dispatch)
        this.computeOrientations(enc, cache.gaussianPyramid, cache.octaveSizes);
    }

    /**
     * Runs the prepare_dispatch shader to compute indirect dispatch parameters
     * based on actual keypoint count.
     */
    runPrepareDispatch(enc) {
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
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    async computeDescriptors(keypoints) {
        const t0 = performance.now();
        if (!this.pyramidCache) throw new Error("Pyramids not allocated. Call detectKeypoints first.");

        await this.uploadKeypoints(keypoints);

        const enc = this.device.createCommandEncoder();

        // Prepare indirect dispatch for the uploaded keypoints
        this.runPrepareDispatch(enc);

        // 2. Compute Descriptors
        const cache = this.pyramidCache;
        this.recordComputeDescriptors(enc, cache.gaussianPyramid, cache.octaveSizes);

        // 3. Readback (Legacy/Direct mode)
        // Since we provided specific keypoints, we might only want to read back 'keypoints.length' descriptors.
        // But our recordReadbackDescriptors copies EVERYTHING (maxKeypoints).
        // For this specific method, we can stick to the old way OR use the new way.
        // Let's use the new way for consistency, but we need to know WHICH staging buffer to use?
        // Actually, let's use a temporary buffer here to avoid messing up the pipelined staging buffers
        // OR just use the pipeline.

        // Using pipeline for consistency:
        this.recordReadbackDescriptors(enc);

        this.device.queue.submit([enc.finish()]);

        this.timings.descriptors = performance.now() - t0;

        // 3. Readback Descriptors
        const allDescriptors = await this.awaitReadbackDescriptors(false); // False because we didn't increment frameIdx in this method

        // Slice to relevant count
        const descriptors = allDescriptors.slice(0, keypoints.length);

        // 4. Merge descriptors into keypoints objects
        for (let i = 0; i < keypoints.length; i++) {
            keypoints[i].descriptor = descriptors[i];
        }

        return keypoints;
    }

    async uploadKeypoints(keypoints) {
        // Format: count (u32), padding (3*u32), [x, y, octave, scale, sigma, ori, pad, pad] (8*f32)
        const count = keypoints.length;
        const countData = new Uint32Array([count, 0, 0, 0]);
        this.device.queue.writeBuffer(this.buffers.keypoints, 0, countData);

        // Pack keypoints
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
            kpData[off + 6] = 0; // pad
            kpData[off + 7] = 0; // pad
        }
        // Offset 16 bytes
        this.device.queue.writeBuffer(this.buffers.keypoints, 16, kpData);
    }

    runGrayscale(enc, outputTex, pw, ph) {
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
        pass.dispatchWorkgroups(Math.ceil(pw / 16), Math.ceil(ph / 16));
        pass.end();
    }

    buildPyramids(enc, pw, ph, cache) {
        let w = pw, h = ph;
        const { gaussianPyramid, dogPyramid, octaveSizes, tempTexture } = cache;

        for (let o = 0; o < this.options.numOctaves; o++) {
            const gaussOctave = gaussianPyramid[o];
            const dogOctave = dogPyramid[o];

            // First Scale
            if (o === 0) {
                this.runBlur(enc, cache.baseTexture, gaussOctave[0], tempTexture, w, h, this.options.sigmaBase);
            } else {
                const prev = gaussianPyramid[o - 1][this.options.scalesPerOctave];
                const prevW = octaveSizes[o - 1].w;
                const prevH = octaveSizes[o - 1].h;
                this.runDownsample(enc, prev, gaussOctave[0], prevW, prevH, w, h);
            }

            // Other Scales
            for (let s = 1; s < gaussOctave.length; s++) {
                const sigma = this.getSigma(s);
                const prevSigma = this.getSigma(s - 1);
                const diff = Math.sqrt(sigma * sigma - prevSigma * prevSigma);
                this.runBlur(enc, gaussOctave[s - 1], gaussOctave[s], tempTexture, w, h, diff);
            }

            // DoG
            for (let s = 0; s < dogOctave.length; s++) {
                this.runDoG(enc, gaussOctave[s], gaussOctave[s + 1], dogOctave[s], w, h);
            }

            w = Math.floor(w / 2); h = Math.floor(h / 2);
        }
    }

    runBlur(enc, inTex, outTex, tempTex, w, h, sigma) {
        // w, h are PACKED dimensions
        const radius = Math.ceil(sigma * 3);
        const kernelBuf = this.getKernelBuffer(sigma, radius); // Cached

        const params = new Uint32Array([w, h, radius, 0]);
        const paramsOffset = this.writeParams(enc, params);

        // Helper for bind/dispatch
        const runPass = (pipeline, inView, outView) => { // Removed tmpView as it's not used in this scope
            const bind = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset: paramsOffset, size: 16 } },
                    { binding: 1, resource: inView },
                    { binding: 2, resource: outView },
                    { binding: 3, resource: { buffer: kernelBuf } }
                ]
            });
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
            pass.end();
        };

        runPass(this.pipelines.blurH, inTex.createView(), tempTex.createView());
        runPass(this.pipelines.blurV, tempTex.createView(), outTex.createView());
    }

    runDoG(enc, texA, texB, outTex, w, h) {
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

    runDownsample(enc, inTex, outTex, sw, sh, dw, dh) {
        const params = new Uint32Array([sw, sh, dw, dh]);
        const paramsOffset = this.writeParams(enc, params);

        const bind = this.device.createBindGroup({
            layout: this.pipelines.downsample.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset: paramsOffset, size: 16 } },
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

    detectExtrema(enc, dogPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o]; // PACKED DIMS

            for (let s = 1; s <= this.options.scalesPerOctave; s++) {
                const params = new ArrayBuffer(24);
                const view = new DataView(params);
                view.setInt32(0, w, true);
                view.setInt32(4, h, true);
                view.setInt32(8, o, true);
                view.setInt32(12, s, true);
                view.setFloat32(16, this.options.contrastThreshold / this.options.scalesPerOctave, true);
                view.setFloat32(20, this.options.edgeThreshold, true); // Edge
                const paramsOffset = this.writeParams(enc, params);

                const bind = this.device.createBindGroup({
                    layout: this.pipelines.extrema.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset: paramsOffset, size: 24 } },
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

    computeOrientations(enc, gaussianPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o];
            const octaveParams = new Int32Array([w, h, o]);
            const paramsOffset = this.writeParams(enc, octaveParams);

            const bind = this.device.createBindGroup({
                layout: this.pipelines.orientation.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset: paramsOffset, size: 12 } },
                    { binding: 1, resource: { buffer: this.buffers.keypoints } },
                    { binding: 2, resource: gaussianPyramid[o][1].createView() },
                    { binding: 3, resource: gaussianPyramid[o][2].createView() },
                    { binding: 4, resource: gaussianPyramid[o][3].createView() }
                ]
            });

            const pass = enc.beginComputePass();
            pass.setPipeline(this.pipelines.orientation);
            pass.setBindGroup(0, bind);
            // Use indirect dispatch: offset 0 = orientation params
            pass.dispatchWorkgroupsIndirect(this.buffers.indirectDispatch, 0);
            pass.end();
        }
    }

    recordComputeDescriptors(enc, gaussianPyramid, sizes) {
        for (let o = 0; o < this.options.numOctaves; o++) {
            const { w, h } = sizes[o];
            const params = new Int32Array([w, h, o, 0]);
            const paramsOffset = this.writeParams(enc, params);

            const bind = this.device.createBindGroup({
                layout: this.pipelines.descriptor.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.unifiedParams, offset: paramsOffset, size: 16 } },
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
            // Use indirect dispatch: offset 12 = descriptor params
            pass.dispatchWorkgroupsIndirect(this.buffers.indirectDispatch, 12);
            pass.end();
        }
    }

    recordReadbackKeypoints(enc) {
        const curIdx = this.frameIdx % this.numStages;
        const curBuf = this.buffers.staging[curIdx];
        enc.copyBufferToBuffer(this.buffers.keypoints, 0, curBuf, 0, curBuf.size);
    }

    triggerKeypointReadback() {
        const curIdx = this.frameIdx % this.numStages;
        const buf = this.buffers.staging[curIdx];
        this.mapPromises[curIdx] = buf.mapAsync(GPUMapMode.READ);
        return this.mapPromises[curIdx];
    }

    triggerDescriptorReadback() {
        const curIdx = this.frameIdx % this.numStages;
        const buf = this.buffers.stagingDescriptors[curIdx];
        this.mapDescPromises[curIdx] = buf.mapAsync(GPUMapMode.READ);
        return this.mapDescPromises[curIdx];
    }

    async retrieveKeypoints(lookback = false) {
        const targetIdx = lookback ? (this.frameIdx - 1) : this.frameIdx;
        if (targetIdx < 0) return [];

        const idx = targetIdx % this.numStages;
        if (!this.mapPromises[idx]) {
            // Auto-trigger map if not already done (useful for one-shot calls)
            this.mapPromises[idx] = this.buffers.staging[idx].mapAsync(GPUMapMode.READ);
        }

        await this.mapPromises[idx];
        const buf = this.buffers.staging[idx];
        const mapped = buf.getMappedRange();
        const count = new Uint32Array(mapped, 0, 1)[0];
        const results = [];

        if (count > 0) {
            const kpData = new Float32Array(mapped, 16, count * 8);
            const factor = this.scaleRestoreFactor;
            for (let i = 0; i < count; i++) {
                const off = i * 8;
                results.push({
                    x: kpData[off + 0] * factor,
                    y: kpData[off + 1] * factor,
                    octave: kpData[off + 2],
                    scale: kpData[off + 3] * factor,
                    sigma: kpData[off + 4],
                    orientation: kpData[off + 5]
                });
            }
        }
        buf.unmap();
        this.mapPromises[idx] = null;
        return results;
    }

    // Legacy support for non-pipelined calls
    async awaitReadbackKeypoints() {
        this.keypoints = await this.retrieveKeypoints(false);
        this.frameIdx++;
    }

    recordReadbackDescriptors(enc) {
        const curIdx = this.frameIdx % this.numStages;
        const curBuf = this.buffers.stagingDescriptors[curIdx];
        enc.copyBufferToBuffer(this.buffers.descriptors, 0, curBuf, 0, curBuf.size);
    }

    async retrieveDescriptors(lookback = false, returnFlat = false) {
        const targetIdx = lookback ? (this.frameIdx - 1) : this.frameIdx;
        if (targetIdx < 0) return returnFlat ? (this.options.quantizeDescriptors ? new Uint8Array(0) : new Float32Array(0)) : [];

        const idx = targetIdx % this.numStages;
        if (!this.mapDescPromises[idx]) {
            // Auto-trigger map if not already done
            this.mapDescPromises[idx] = this.buffers.stagingDescriptors[idx].mapAsync(GPUMapMode.READ);
        }

        await this.mapDescPromises[idx];
        const buf = this.buffers.stagingDescriptors[idx];
        const mapped = buf.getMappedRange();

        const isQuant = this.options.quantizeDescriptors;
        const result = isQuant ? new Uint8Array(mapped).slice() : new Float32Array(mapped).slice();
        buf.unmap();
        this.mapDescPromises[idx] = null;

        // Determine how many descriptors to return. 
        // In one-shot mode, we know this from this.keypoints.
        // In pipelined mode, if returnFlat is used, the caller might handle slicing.
        const count = (this.keypoints && this.keypoints.length > 0) ? this.keypoints.length : Math.floor(result.length / 128);

        if (returnFlat) return result.subarray(0, count * 128).slice();

        const descriptors = new Array(count);
        for (let i = 0; i < count; i++) {
            descriptors[i] = result.slice(i * 128, (i + 1) * 128);
        }
        return descriptors;
    }

    async awaitReadbackDescriptors(useLookback = false, returnFlat = false) {
        return this.retrieveDescriptors(useLookback, returnFlat);
    }

}
