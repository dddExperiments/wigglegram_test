// import { MAX_KEYPOINTS, NUM_OCTAVES, SCALES_PER_OCTAVE, SIGMA_BASE, CONTRAST_THRESHOLD } from './constants.js';
import { SIFT } from './sift.js';
import { TexturePool } from './utils/texture-pool.js';

export class SIFTWebGPUBase extends SIFT {
    constructor(options) {
        super(options);
        this.device = null;
        this.shaderCache = new Map();
        this.kernelCache = new Map();
        this.pipelines = {};
        this.texturePool = null;
    }

    /**
     * Optional: Enable texture pooling to reduce allocation overhead 
     * when processing many images of different/same resolutions.
     */
    enableTexturePool(poolSize = 10) {
        if (!this.device) throw new Error("Initialize SIFT (init) before enabling pool");
        this.texturePool = new TexturePool(this.device, { maxPoolSize: poolSize });
    }

    async init() {
        if (!navigator.gpu) throw new Error("WebGPU not supported");

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error("No WebGPU adapter found");

        // Request maximal limits for high-resolution image processing, capped at 1GB
        const oneGB = 1024 * 1024 * 1024;
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: Math.min(adapter.limits.maxBufferSize, oneGB),
                maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, oneGB),
                maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
            }
        });
        this.log(`[WebGPU] Device initialized with max buffer size: ${this.device.limits.maxBufferSize} bytes`);

        // Pre-compute kernels for blur operations
        this.precomputeKernels();
    }

    /**
     * Pre-computes all Gaussian kernels used in the pyramid construction.
     * SIFT kernels are deterministic based on sigmaBase and scalesPerOctave.
     */
    precomputeKernels() {
        // 1. Base sigma kernel
        const sigmaBase = this.options.sigmaBase;
        this.getKernelBuffer(sigmaBase, Math.ceil(sigmaBase * 3));

        // 2. Incremental blur kernels for other scales
        // GaussOctave has scales s=0...scalesPerOctave+2
        for (let s = 1; s <= this.options.scalesPerOctave + 2; s++) {
            const sigma = this.getSigma(s);
            const prevSigma = this.getSigma(s - 1);
            const diff = Math.sqrt(sigma * sigma - prevSigma * prevSigma);
            this.getKernelBuffer(diff, Math.ceil(diff * 3));
        }
        this.log(`[SIFT] Pre-computed ${this.options.scalesPerOctave + 3} Gaussian kernels`);
    }

    async initPipelines(basePath) {
        const getPath = (name, defaultFile) => `${basePath}/${defaultFile}`;

        this.pipelines.grayscale = this.createComputePipeline('grayscale', await this.loadShader(getPath('grayscale', 'grayscale.wgsl')));
        this.pipelines.blurH = this.createComputePipeline('blurH', await this.loadShader(getPath('blurH', 'blur_horizontal.wgsl')));
        this.pipelines.blurV = this.createComputePipeline('blurV', await this.loadShader(getPath('blurV', 'blur_vertical.wgsl')));
        this.pipelines.dog = this.createComputePipeline('dog', await this.loadShader(getPath('dog', 'dog.wgsl')));
        this.pipelines.downsample = this.createComputePipeline('downsample', await this.loadShader(getPath('downsample', 'downsample.wgsl')));
        this.pipelines.extrema = this.createComputePipeline('extrema', await this.loadShader(getPath('extrema', 'extrema.wgsl')));
        this.pipelines.orientation = this.createComputePipeline('orientation', await this.loadShader(getPath('orientation', 'orientation.wgsl')));
        this.pipelines.descriptor = this.createComputePipeline('descriptor', await this.loadShader(getPath('descriptor', 'descriptor.wgsl')));
    }

    /**
     * Loads and processes an image source into an ImageBitmap, 
     * handling downsampling if configured.
     * @param {string|HTMLImageElement|HTMLVideoElement|ImageBitmap} source 
     * @returns {Promise<ImageBitmap>}
     */
    async processBitmap(source) {
        let bitmap;
        if (typeof source === 'string') {
            const res = await fetch(source);
            const blob = await res.blob();
            bitmap = await createImageBitmap(blob);
        } else if (source instanceof HTMLImageElement || source instanceof HTMLVideoElement) {
            bitmap = await createImageBitmap(source);
        } else {
            bitmap = source;
        }

        this.originalWidth = bitmap.width;
        this.originalHeight = bitmap.height;
        this.scaleRestoreFactor = 1.0;

        const maxDim = this.options.maxImageDimension;
        if (maxDim > 0 && (this.originalWidth > maxDim || this.originalHeight > maxDim)) {
            const scale = Math.min(maxDim / this.originalWidth, maxDim / this.originalHeight);
            const targetWidth = Math.round(this.originalWidth * scale);
            const targetHeight = Math.round(this.originalHeight * scale);

            this.log(`[SIFT] Downsampling: ${this.originalWidth}x${this.originalHeight} -> ${targetWidth}x${targetHeight}`);

            const canvas = new OffscreenCanvas(targetWidth, targetHeight);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
            bitmap = canvas.transferToImageBitmap();

            this.scaleRestoreFactor = 1.0 / scale;
        }
        return bitmap;
    }

    /**
     * Uploads a bitmap to the GPU input texture. 
     * Allocates or resizes the input texture if needed.
     * @param {ImageBitmap} bitmap 
     */
    uploadImageToTexture(bitmap) {
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
        const format = 'rgba8unorm';

        this.width = bitmap.width;
        this.height = bitmap.height;

        if (this.texturePool) {
            // Release old one if it was from pool
            if (this.inputTexture && this.inputTextureFromPool) {
                this.texturePool.release(this.inputTexture);
            }
            this.inputTexture = this.texturePool.acquire(this.width, this.height, format, usage);
            this.inputTextureFromPool = true;
        } else {
            if (!this.inputTexture || this.inputTexture.width !== this.width || this.inputTexture.height !== this.height) {
                if (this.inputTexture) this.inputTexture.destroy();
                this.inputTexture = this.device.createTexture({
                    size: [this.width, this.height],
                    format: format,
                    usage: usage
                });
            }
            this.inputTextureFromPool = false;
        }

        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.inputTexture },
            [this.width, this.height]
        );
    }

    /**
     * Full image loading flow (process + upload)
     * @param {string|HTMLImageElement|HTMLVideoElement|ImageBitmap} source 
     */
    async loadImage(source) {
        const bitmap = await this.processBitmap(source);
        this.uploadImageToTexture(bitmap);
    }

    async loadShader(path) {
        if (this.shaderCache.has(path)) return this.shaderCache.get(path);

        const res = await fetch(path);
        if (!res.ok) throw new Error(`Failed to load shader: ${path}`);
        let code = await res.text();

        code = await processShaderIncludes(path, code);

        const module = this.device.createShaderModule({
            label: path,
            code: code
        });

        this.shaderCache.set(path, module);
        return module;
    }

    createComputePipeline(label, module, entryPoint = 'main', layout = 'auto') {
        return this.device.createComputePipeline({
            label,
            layout,
            compute: { module, entryPoint }
        });
    }

    createStorageTexture(width, height, format = 'rgba32float') {
        return this.device.createTexture({
            size: [width, height],
            format: format,
            usage: GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.COPY_DST
        });
    }

    createBuffer(size, usage, label = '') {
        return this.device.createBuffer({
            label,
            size,
            usage
        });
    }

    /**
     * Gets or creates a cached GPU buffer for a Gaussian kernel.
     * @param {number} sigma 
     * @param {number} radius 
     * @returns {GPUBuffer}
     */
    getKernelBuffer(sigma, radius) {
        const key = `${sigma.toFixed(4)}_${radius}`;
        if (this.kernelCache.has(key)) return this.kernelCache.get(key);

        const kernel = this.createKernel(sigma, radius);
        const buffer = this.createBuffer(kernel.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, `Kernel_${key}`);
        this.device.queue.writeBuffer(buffer, 0, kernel);

        this.kernelCache.set(key, buffer);
        return buffer;
    }

    /**
     * Utility to read back a texture to CPU. 
     * Useful for visualization and debugging.
     * @param {GPUTexture} texture 
     * @returns {Promise<Float32Array>}
     */
    async readbackTexture(texture) {
        const { width, height, format } = texture;
        const isFloat = format && format.toLowerCase().includes('float');
        const bytesPerPixel = isFloat ? 16 : 4;

        const unalignedBytesPerRow = width * bytesPerPixel;
        const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
        const size = bytesPerRow * height;

        const readBuffer = this.createBuffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = this.device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture }, { buffer: readBuffer, bytesPerRow }, [width, height]);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = readBuffer.getMappedRange();

        const bytesPerElement = isFloat ? 4 : 1;
        const elementsPerRow = bytesPerRow / bytesPerElement;
        const unalignedElementsPerRow = unalignedBytesPerRow / bytesPerElement;
        const fullData = new (isFloat ? Float32Array : Uint8Array)(mappedRange);

        const result = new (isFloat ? Float32Array : Uint8Array)(width * height * 4);

        let maxVal = -Infinity;
        let minVal = Infinity;

        for (let y = 0; y < height; y++) {
            const srcOffset = y * elementsPerRow;
            const dstOffset = y * unalignedElementsPerRow;
            const rowData = fullData.subarray(srcOffset, srcOffset + unalignedElementsPerRow);
            result.set(rowData, dstOffset);

            // Debug: Check stats
            for (let i = 0; i < rowData.length; i++) {
                if (rowData[i] > maxVal) maxVal = rowData[i];
                if (rowData[i] < minVal) minVal = rowData[i];
            }
        }

        this.log(`[Readback] ${width}x${height} ${format} | Min: ${minVal.toFixed(4)} Max: ${maxVal.toFixed(4)}`);

        readBuffer.unmap();
        readBuffer.destroy();
        return result;
    }
}

/**
 * Processes #include "..." directives in WGSL code.
 * @param {string} path - Path of the current shader file
 * @param {string} code - WGSL code
 * @returns {Promise<string>} - Code with includes resolved
 */
export async function processShaderIncludes(path, code) {
    const includeRegex = /#include\s+"([^"]+)"/;
    let match;
    while ((match = code.match(includeRegex)) !== null) {
        const includeDirective = match[0];
        const includePath = match[1];

        // Resolve relative to current shader path
        const parts = path.split('/');
        parts.pop();
        const includeParts = includePath.split('/');
        for (const p of includeParts) {
            if (p === '..') parts.pop();
            else if (p !== '.') parts.push(p);
        }
        const fullIncludePath = parts.join('/');

        try {
            const incRes = await fetch(fullIncludePath);
            if (incRes.ok) {
                const incCode = await incRes.text();
                code = code.replace(includeDirective, incCode);
            } else {
                console.warn(`Could not load include: ${fullIncludePath}`);
                code = code.replace(includeDirective, `// Failed to include: ${includePath}`);
            }
        } catch (e) {
            console.warn(`Error loading include: ${fullIncludePath}`, e);
            code = code.replace(includeDirective, `// Error including: ${includePath}`);
        }
    }
    return code;
}

