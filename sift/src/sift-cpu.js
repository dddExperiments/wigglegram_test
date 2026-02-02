import { SIFT } from './sift.js';

/**
 * CPU Implementation of SIFT
 */
export class SIFTCPU extends SIFT {
    constructor(options) {
        super(options);
        this.imageData = null;
        this.grayData = null;
        this.gaussianPyramid = [];
        this.dogPyramid = [];
        this.octaveSizes = [];
    }

    async loadImage(source) {
        if (typeof source === 'string') {
            await this.loadImageFromUrl(source);
        } else if (source instanceof ImageBitmap || source instanceof HTMLImageElement || source instanceof HTMLCanvasElement) {
            this.processImage(source);
        }
    }

    async loadImageFromUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                this.processImage(img);
                resolve();
            };
            img.onerror = reject;
            img.src = url;
        });
    }

    processImage(img) {
        this.width = img.width;
        this.height = img.height;

        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        this.imageData = ctx.getImageData(0, 0, this.width, this.height);
    }

    async detectKeypoints(image) {
        const t0 = performance.now();
        this.timings = {};

        // Load/Process Image
        await this.loadImage(image);

        // 1. Grayscale
        this.computeGrayscale();
        const t1 = performance.now();
        this.timings.grayscale = t1 - t0;

        // 2. Gaussian Pyramid
        this.buildGaussianPyramid();
        const t2 = performance.now();
        this.timings.gaussian = t2 - t1;

        // 3. DoG Pyramid
        this.computeDoGPyramid();
        const t3 = performance.now();
        this.timings.dog = t3 - t2;

        // 4. Extrema
        this.detectExtrema();
        const t4 = performance.now();
        this.timings.extrema = t4 - t3;

        // 5. Orientations
        this.computeOrientations();
        const t5 = performance.now();
        this.timings.orientation = t5 - t4;

        this.timings.detection = t5 - t0;
        return this.keypoints;
    }

    async computeDescriptors(keypoints) {
        const t0 = performance.now();

        // Use provided keypoints if any (overwrite internal state for descriptor computation)
        if (keypoints) {
            this.keypoints = keypoints;
        }

        // 6. Descriptors
        this.computeDescriptorsImpl();
        const t6 = performance.now();
        this.timings.descriptors = t6 - t0;

        return this.keypoints;
    }

    // Legacy run() wrapper if needed, but base class has detectSift
    // We can remove run() as base class detectSift calls detectKeypoints + computeDescriptors

    // Rename internal computeDescriptors to computeDescriptorsImpl to avoid conflict
    computeDescriptorsImpl() {
        for (const kp of this.keypoints) {
            const o = kp.octave;
            const gaussian = this.gaussianPyramid[o][kp.scale];
            const { w, h } = this.octaveSizes[o];
            const scale = Math.pow(2, o);

            const localX = kp.x / scale;
            const localY = kp.y / scale;
            const angle = kp.orientation;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);

            // New scale-dependent step size (match GPU)
            const localSigma = kp.sigma / scale;
            const step = 0.75 * localSigma;

            const hist = new Float32Array(128);

            for (let r = -8; r < 8; r++) {
                for (let c = -8; c < 8; c++) {
                    const rotX = step * (c * cos - r * sin);
                    const rotY = step * (c * sin + r * cos);

                    const x = localX + rotX;
                    const y = localY + rotY;

                    const ix = Math.floor(x);
                    const iy = Math.floor(y);

                    if (ix < 1 || ix >= w - 2 || iy < 1 || iy >= h - 2) continue;

                    const sx = Math.round(x);
                    const sy = Math.round(y);

                    if (sx < 1 || sx >= w - 1 || sy < 1 || sy >= h - 1) continue;

                    const idx = sy * w + sx;
                    const gx = gaussian[idx + 1] - gaussian[idx - 1];
                    const gy = gaussian[idx + w] - gaussian[idx - w];
                    const mag = Math.sqrt(gx * gx + gy * gy);
                    let ori = Math.atan2(gy, gx) - angle;
                    while (ori < 0) ori += 2 * Math.PI;
                    while (ori >= 2 * Math.PI) ori -= 2 * Math.PI;

                    const rbin = (r + 8) / 4 - 0.5;
                    const cbin = (c + 8) / 4 - 0.5;
                    const obin = (ori * 8) / (2 * Math.PI);

                    const weight = mag * Math.exp(-(r * r + c * c) / 32);

                    for (let dr = 0; dr < 2; dr++) {
                        const ri = Math.floor(rbin) + dr;
                        if (ri < 0 || ri >= 4) continue;
                        const rw = dr === 0 ? (1 - (rbin - Math.floor(rbin))) : (rbin - Math.floor(rbin));

                        for (let dc = 0; dc < 2; dc++) {
                            const ci = Math.floor(cbin) + dc;
                            if (ci < 0 || ci >= 4) continue;
                            const cw = dc === 0 ? (1 - (cbin - Math.floor(cbin))) : (cbin - Math.floor(cbin));

                            for (let do_idx = 0; do_idx < 2; do_idx++) {
                                const oi_raw = Math.floor(obin) + do_idx;
                                const ow = do_idx === 0 ? (1 - (obin - Math.floor(obin))) : (obin - Math.floor(obin));

                                const oi = (oi_raw + 8) % 8;
                                const binIdx = (ri * 4 + ci) * 8 + oi;

                                hist[binIdx] += weight * rw * cw * ow;
                            }
                        }
                    }
                }
            }

            let norm = 0;
            for (let i = 0; i < 128; i++) norm += hist[i] * hist[i];
            norm = Math.sqrt(norm) + 1e-7;
            for (let i = 0; i < 128; i++) hist[i] = Math.min(0.2, hist[i] / norm);

            norm = 0;
            for (let i = 0; i < 128; i++) norm += hist[i] * hist[i];
            norm = Math.sqrt(norm) + 1e-7;
            for (let i = 0; i < 128; i++) hist[i] /= norm;

            kp.descriptor = hist;
        }
    }

    computeGrayscale() {
        const data = this.imageData.data;
        const n = this.width * this.height;
        this.grayData = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const r = data[i * 4] / 255;
            const g = data[i * 4 + 1] / 255;
            const b = data[i * 4 + 2] / 255;
            this.grayData[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }

    buildGaussianPyramid() {
        this.gaussianPyramid = [];
        this.octaveSizes = [];
        let w = this.width;
        let h = this.height;
        let base = this.grayData;

        for (let o = 0; o < this.options.numOctaves; o++) {
            this.octaveSizes.push({ w, h });
            const octave = [];

            // Base for this octave
            if (o === 0) {
                octave.push(this.gaussianBlur(base, w, h, this.options.sigmaBase));
            } else {
                // Downsample from previous octave (scale index SCALES_PER_OCTAVE)
                const prevOctave = this.gaussianPyramid[o - 1];
                const prevW = this.octaveSizes[o - 1].w;
                const prevH = this.octaveSizes[o - 1].h;
                const down = this.downsample(prevOctave[this.options.scalesPerOctave], prevW, prevH);
                w = down.w;
                h = down.h;
                this.octaveSizes[o] = { w, h };
                octave.push(down.data);
            }

            // Remaining scales
            for (let s = 1; s < this.options.scalesPerOctave + 3; s++) {
                const sigma = this.getSigma(s);
                const prevSigma = this.getSigma(s - 1);
                const sigmaDiff = Math.sqrt(sigma * sigma - prevSigma * prevSigma);
                octave.push(this.gaussianBlur(octave[s - 1], w, h, sigmaDiff));
            }

            this.gaussianPyramid.push(octave);
        }
    }


    computeDoGPyramid() {
        this.dogPyramid = [];
        for (let o = 0; o < this.options.numOctaves; o++) {
            const gaussian = this.gaussianPyramid[o];
            const { w, h } = this.octaveSizes[o];
            const scaleDoG = [];
            for (let s = 0; s < gaussian.length - 1; s++) {
                const dog = new Float32Array(w * h);
                const g1 = gaussian[s + 1];
                const g0 = gaussian[s];
                for (let i = 0; i < w * h; i++) {
                    dog[i] = g1[i] - g0[i];
                }
                scaleDoG.push(dog);
            }
            this.dogPyramid.push(scaleDoG);
        }
    }

    detectExtrema() {
        this.keypoints = [];
        for (let o = 0; o < this.options.numOctaves; o++) {
            const dog = this.dogPyramid[o];
            const { w, h } = this.octaveSizes[o];
            const scaleMult = Math.pow(2, o);

            for (let s = 1; s <= this.options.scalesPerOctave; s++) {
                const prev = dog[s - 1];
                const curr = dog[s];
                const next = dog[s + 1];

                for (let y = 1; y < h - 1; y++) {
                    for (let x = 1; x < w - 1; x++) {
                        const idx = y * w + x;
                        const val = curr[idx];

                        if (Math.abs(val) < this.options.contrastThreshold / this.options.scalesPerOctave) continue;

                        if (this.isExtremum(val, x, y, w, prev, curr, next)) {
                            if (!this.isEdge(val, x, y, w, curr)) {
                                this.keypoints.push({
                                    x: x * scaleMult,
                                    y: y * scaleMult,
                                    octave: o,
                                    scale: s,
                                    sigma: this.getSigma(s) * scaleMult,
                                    orientation: 0
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    isExtremum(val, x, y, w, prev, curr, next) {
        let isMax = true;
        let isMin = true;
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const idx = (y + dy) * w + (x + dx);
                if (prev[idx] >= val) isMax = false;
                if (prev[idx] <= val) isMin = false;
                if (dx !== 0 || dy !== 0) {
                    if (curr[idx] >= val) isMax = false;
                    if (curr[idx] <= val) isMin = false;
                }
                if (next[idx] >= val) isMax = false;
                if (next[idx] <= val) isMin = false;
                if (!isMax && !isMin) return false;
            }
        }
        return isMax || isMin;
    }

    isEdge(val, x, y, w, data) {
        const idx = y * w + x;
        const dxx = data[idx + 1] + data[idx - 1] - 2 * val;
        const dyy = data[idx + w] + data[idx - w] - 2 * val;
        const dxy = (data[idx + w + 1] - data[idx + w - 1] - data[idx - w + 1] + data[idx - w - 1]) / 4;

        // Edge check: Reject points that have a large principal curvature in one direction
        // but a small one in the other (edges).
        // Uses the ratio of eigenvalues of the 2x2 Hessian matrix.
        const tr = dxx + dyy;
        const det = dxx * dyy - dxy * dxy;
        const r = this.options.edgeThreshold;

        if (det <= 0) return true;
        return (tr * tr * r) >= (r + 1) * (r + 1) * det;
    }

    computeOrientations() {
        for (const kp of this.keypoints) {
            const o = kp.octave;
            const gaussian = this.gaussianPyramid[o][kp.scale];
            const { w, h } = this.octaveSizes[o];
            const scale = Math.pow(2, o);

            const localX = Math.round(kp.x / scale);
            const localY = Math.round(kp.y / scale);
            const localSigma = kp.sigma / scale;
            const radius = Math.round(localSigma * 1.5 * 3); // 3 * 1.5 * sigma radius (to match standard)

            const numBins = 36;
            const hist = new Float32Array(numBins);

            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    const x = localX + dx;
                    const y = localY + dy;
                    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) continue;

                    const idx = y * w + x;
                    const gx = gaussian[idx + 1] - gaussian[idx - 1];
                    const gy = gaussian[idx + w] - gaussian[idx - w];
                    const mag = Math.sqrt(gx * gx + gy * gy);
                    let ang = Math.atan2(gy, gx) * 180 / Math.PI;
                    if (ang < 0) ang += 360;

                    const bin = Math.floor(ang * numBins / 360) % numBins;
                    const sigmaW = 1.5 * localSigma;
                    const weight = Math.exp(-(dx * dx + dy * dy) / (2 * sigmaW * sigmaW));
                    hist[bin] += mag * weight;
                }
            }

            // Smooth Histogram [0.25, 0.5, 0.25]
            const smoothedHist = new Float32Array(numBins);
            for (let i = 0; i < numBins; i++) {
                const prev = hist[(i + numBins - 1) % numBins];
                const curr = hist[i];
                const next = hist[(i + 1) % numBins];
                smoothedHist[i] = 0.25 * prev + 0.5 * curr + 0.25 * next;
            }

            // Peak finding
            let maxVal = 0;
            let maxBin = -1;
            for (let i = 0; i < numBins; i++) {
                if (smoothedHist[i] > maxVal) { maxVal = smoothedHist[i]; maxBin = i; }
            }

            // Parabolic interpolation
            const left = smoothedHist[(maxBin + numBins - 1) % numBins];
            const right = smoothedHist[(maxBin + 1) % numBins];
            const peak = maxBin + 0.5 * (left - right) / (left - 2 * maxVal + right);
            kp.orientation = peak * 360 / numBins * Math.PI / 180;
        }
    }



    gaussianBlur(src, w, h, sigma) {
        const radius = Math.ceil(sigma * 3);
        const kernel = this.createKernel(sigma, radius);

        const temp = new Float32Array(w * h);
        const dst = new Float32Array(w * h);

        // H pass
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let val = 0;
                for (let i = -radius; i <= radius; i++) {
                    const sx = Math.min(Math.max(x + i, 0), w - 1);
                    val += src[y * w + sx] * kernel[i + radius];
                }
                temp[y * w + x] = val;
            }
        }

        // V pass
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let val = 0;
                for (let i = -radius; i <= radius; i++) {
                    const sy = Math.min(Math.max(y + i, 0), h - 1);
                    val += temp[sy * w + x] * kernel[i + radius];
                }
                dst[y * w + x] = val;
            }
        }
        return dst;
    }

    downsample(src, w, h) {
        const halfW = Math.floor(w / 2);
        const halfH = Math.floor(h / 2);
        const dst = new Float32Array(halfW * halfH);
        for (let y = 0; y < halfH; y++) {
            for (let x = 0; x < halfW; x++) {
                dst[y * halfW + x] = src[(y * 2) * w + (x * 2)];
            }
        }
        return { data: dst, w: halfW, h: halfH };
    }
}
