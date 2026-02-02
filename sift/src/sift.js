/**
 * SIFT Base Class
 * Defines the interface for SIFT implementations (CPU, WebGPU, etc.)
 */
export class SIFT {
    constructor(options = {}) {
        this.options = {
            numOctaves: 4,
            scalesPerOctave: 3,
            sigmaBase: 1.6,
            contrastThreshold: 0.03,
            edgeThreshold: 10,
            maxKeypoints: 100000,
            maxImageDimension: 3000, // 0 = no limit
            quantizeDescriptors: false,
            debug: false,
            ...options
        };
        this.width = 0;
        this.height = 0;
        this.originalWidth = 0;
        this.originalHeight = 0;
        this.scaleRestoreFactor = 1.0;
        this.keypoints = [];
        this.timings = {};
    }

    /**
     * Initialize resources (if any)
     */
    async init() { }

    /**
     * Detect keypoints in the image
     * @param {string|ImageBitmap|HTMLImageElement} image 
     * @returns {Promise<Array>} List of keypoints (without descriptors)
     */
    async detectKeypoints(image) {
        throw new Error("detectKeypoints must be implemented");
    }

    /**
     * Compute descriptors for the given keypoints
     * @param {Array} keypoints List of keypoints
     * @returns {Promise<Array>} List of keypoints with descriptors
     */
    async computeDescriptors(keypoints) {
        throw new Error("computeDescriptors must be implemented");
    }

    /**
     * Run full SIFT pipeline (detect + describe)
     * @param {string|ImageBitmap|HTMLImageElement} image 
     * @returns {Promise<Array>} List of keypoints with descriptors
     */
    async detectAndCompute(image) {
        const keypoints = await this.detectKeypoints(image);
        return this.computeDescriptors(keypoints);
    }

    /**
     * @deprecated Use detectAndCompute instead
     */
    async detectSift(image) {
        return this.detectAndCompute(image);
    }

    /**
     * Load image helper (internal use mostly)
     * @param {string|ImageBitmap} source 
    */
    async ensureImage(source) {
        if (typeof source === 'string') {
            const res = await fetch(source);
            const blob = await res.blob();
            return createImageBitmap(blob);
        } else if (source instanceof HTMLImageElement || source instanceof HTMLVideoElement) {
            return createImageBitmap(source);
        } else {
            return source;
        }
    }

    getFeatureCount() {
        return this.keypoints.length;
    }

    getTimings() {
        return this.timings;
    }

    getSigma(s) {
        return this.options.sigmaBase * Math.pow(2.0, s / this.options.scalesPerOctave);
    }

    /**
     * Creates a 1D Gaussian kernel.
     * @param {number} sigma 
     * @param {number} radius 
     * @returns {Float32Array}
     */
    createKernel(sigma, radius) {
        const size = radius * 2 + 1;
        const kernel = new Float32Array(size);
        let sum = 0;
        for (let i = -radius; i <= radius; i++) {
            const v = Math.exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + radius] = v;
            sum += v;
        }
        for (let i = 0; i < size; i++) kernel[i] /= sum;
        return kernel;
    }

    log(...args) {
        if (this.options.debug) {
            console.log(...args);
        }
    }
}
