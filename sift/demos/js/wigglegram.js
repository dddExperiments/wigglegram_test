/**
 * Browser-based Wigglegram Generator
 * 
 * Pure JavaScript implementation of:
 * - RANSAC homography estimation
 * - Perspective warping
 * - GIF encoding
 */

import { Matrix } from '../../src/math/matrix.js';
import { HomographyMatrixRANSAC } from '../../src/geometry/homography_matrix.js';

// ============================================================================
// Image Warping
// ============================================================================

class ImageWarp {
    /**
     * Map a point using a homography matrix
     */
    static transformPoint(H, x, y) {
        const w = H[2][0] * x + H[2][1] * y + H[2][2];
        const dstX = (H[0][0] * x + H[0][1] * y + H[0][2]) / w;
        const dstY = (H[1][0] * x + H[1][1] * y + H[1][2]) / w;
        return [dstX, dstY];
    }

    /**
     * Warp an image using a homography with bilinear interpolation
     * @param {ImageData} srcImg - Source image data
     * @param {Array} H - 3x3 homography matrix
     * @param {number} dstWidth - Output width
     * @param {number} dstHeight - Output height
     * @returns {ImageData} Warped image
     */
    static perspectiveWarp(srcImg, H, dstWidth, dstHeight) {
        const srcData = srcImg.data;
        const srcWidth = srcImg.width;
        const srcHeight = srcImg.height;

        // Invert H to map dst -> src
        const Hinv = Matrix.invert3x3(H);
        if (!Hinv) {
            console.error('Failed to invert homography');
            return srcImg;
        }

        // Create output image
        const dstData = new Uint8ClampedArray(dstWidth * dstHeight * 4);

        for (let dstY = 0; dstY < dstHeight; dstY++) {
            for (let dstX = 0; dstX < dstWidth; dstX++) {
                // Map destination to source using inverse H
                const [srcX, srcY] = ImageWarp.transformPoint(Hinv, dstX, dstY);

                // Bilinear interpolation
                const dstIdx = (dstY * dstWidth + dstX) * 4;

                if (srcX >= 0 && srcX < srcWidth - 1 && srcY >= 0 && srcY < srcHeight - 1) {
                    const x0 = Math.floor(srcX);
                    const y0 = Math.floor(srcY);
                    const x1 = x0 + 1;
                    const y1 = y0 + 1;

                    const fx = srcX - x0;
                    const fy = srcY - y0;

                    const w00 = (1 - fx) * (1 - fy);
                    const w10 = fx * (1 - fy);
                    const w01 = (1 - fx) * fy;
                    const w11 = fx * fy;

                    const idx00 = (y0 * srcWidth + x0) * 4;
                    const idx10 = (y0 * srcWidth + x1) * 4;
                    const idx01 = (y1 * srcWidth + x0) * 4;
                    const idx11 = (y1 * srcWidth + x1) * 4;

                    for (let c = 0; c < 4; c++) {
                        dstData[dstIdx + c] =
                            w00 * srcData[idx00 + c] +
                            w10 * srcData[idx10 + c] +
                            w01 * srcData[idx01 + c] +
                            w11 * srcData[idx11 + c];
                    }
                } else {
                    // Out of bounds - transparent black
                    dstData[dstIdx + 3] = 0;
                }
            }
        }

        return new ImageData(dstData, dstWidth, dstHeight);
    }
}

// ============================================================================
// GIF Encoder (based on gif.js approach)
// ============================================================================

class GIFEncoder {
    /**
     * Create an animated GIF from frames
     * @param {Array<ImageData>} frames - Array of ImageData objects
     * @param {number} delay - Frame delay in 1/100th seconds (e.g., 20 = 200ms)
     * @returns {Uint8Array} GIF binary data
     */
    static encode(frames, delay = 20) {
        if (frames.length === 0) return new Uint8Array(0);

        const width = frames[0].width;
        const height = frames[0].height;

        const output = [];

        // GIF Header
        output.push(...GIFEncoder.stringToBytes('GIF89a'));

        // Logical Screen Descriptor
        output.push(width & 0xFF, (width >> 8) & 0xFF);
        output.push(height & 0xFF, (height >> 8) & 0xFF);
        output.push(0xF6); // Global color table flag + color resolution + sort flag + size (128 colors)
        output.push(0); // Background color index
        output.push(0); // Pixel aspect ratio

        // Global Color Table (128 RGB entries = 384 bytes)
        const colorTable = GIFEncoder.createColorTable();
        output.push(...colorTable);

        // Netscape extension for looping
        output.push(0x21, 0xFF, 0x0B);
        output.push(...GIFEncoder.stringToBytes('NETSCAPE2.0'));
        output.push(0x03, 0x01, 0x00, 0x00, 0x00);

        // Add each frame
        for (const frame of frames) {
            // Graphic Control Extension
            output.push(0x21, 0xF9, 0x04);
            output.push(0x04); // Dispose: restore to background
            output.push(delay & 0xFF, (delay >> 8) & 0xFF); // Delay
            output.push(0x00); // Transparent color index
            output.push(0x00); // Block terminator

            // Image Descriptor
            output.push(0x2C);
            output.push(0x00, 0x00); // Left
            output.push(0x00, 0x00); // Top
            output.push(width & 0xFF, (width >> 8) & 0xFF);
            output.push(height & 0xFF, (height >> 8) & 0xFF);
            output.push(0x00); // No local color table

            // Image Data (LZW compressed)
            const indexedPixels = GIFEncoder.quantizeFrame(frame.data, width, height, colorTable);
            const lzwData = GIFEncoder.lzwEncode(indexedPixels, 7); // Min code size = 7 for 128 colors
            output.push(7); // Min code size
            output.push(...lzwData);
            output.push(0x00); // Block terminator
        }

        // GIF Trailer
        output.push(0x3B);

        return new Uint8Array(output);
    }

    static stringToBytes(str) {
        return str.split('').map(c => c.charCodeAt(0));
    }

    static createColorTable() {
        // Simple 128-color palette (uniform RGB quantization)
        const table = [];
        for (let i = 0; i < 128; i++) {
            const r = ((i >> 4) & 0x3) * 85;
            const g = ((i >> 2) & 0x3) * 85;
            const b = (i & 0x3) * 85;
            table.push(r, g, b);
        }
        // Pad to 384 bytes (128 * 3)
        return table;
    }

    static quantizeFrame(data, width, height, colorTable) {
        const indexed = new Uint8Array(width * height);
        for (let i = 0; i < width * height; i++) {
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];

            // Map to 128-color palette index
            const ri = Math.min(3, Math.floor(r / 64));
            const gi = Math.min(3, Math.floor(g / 64));
            const bi = Math.min(3, Math.floor(b / 64));
            indexed[i] = (ri << 4) | (gi << 2) | bi;
        }
        return indexed;
    }

    static lzwEncode(data, minCodeSize) {
        const clearCode = 1 << minCodeSize;
        const eoiCode = clearCode + 1;

        const output = [];
        let codeSize = minCodeSize + 1;
        let nextCode = eoiCode + 1;
        const dictionary = new Map();

        // Initialize dictionary with single-character strings
        for (let i = 0; i < clearCode; i++) {
            dictionary.set(String.fromCharCode(i), i);
        }

        let bits = 0;
        let bitCount = 0;
        let subBlock = [];

        const writeBits = (code, size) => {
            bits |= (code << bitCount);
            bitCount += size;

            while (bitCount >= 8) {
                subBlock.push(bits & 0xFF);
                bits >>= 8;
                bitCount -= 8;

                if (subBlock.length === 255) {
                    output.push(255);
                    output.push(...subBlock);
                    subBlock = [];
                }
            }
        };

        const flushBits = () => {
            if (bitCount > 0) {
                subBlock.push(bits & 0xFF);
            }
            if (subBlock.length > 0) {
                output.push(subBlock.length);
                output.push(...subBlock);
            }
        };

        writeBits(clearCode, codeSize);

        let current = String.fromCharCode(data[0]);

        for (let i = 1; i < data.length; i++) {
            const next = String.fromCharCode(data[i]);
            const combined = current + next;

            if (dictionary.has(combined)) {
                current = combined;
            } else {
                writeBits(dictionary.get(current), codeSize);

                if (nextCode < 4096) {
                    dictionary.set(combined, nextCode++);
                    if (nextCode > (1 << codeSize) && codeSize < 12) {
                        codeSize++;
                    }
                }

                current = next;
            }
        }

        writeBits(dictionary.get(current), codeSize);
        writeBits(eoiCode, codeSize);
        flushBits();

        return output;
    }
}

// ============================================================================
// Wigglegram Generator
// ============================================================================

class WigglegramGenerator {
    /**
     * Create a wigglegram from two images and their matched keypoints
     * @param {HTMLImageElement|ImageBitmap} imgA - First image
     * @param {HTMLImageElement|ImageBitmap} imgB - Second image
     * @param {Array} keypointsA - Keypoints from first image [{x, y, descriptor}, ...]
     * @param {Array} keypointsB - Keypoints from second image [{x, y, descriptor}, ...]
     * @param {Array} matches - Match pairs [[idxA, idxB], ...]
     * @param {Object} options - Optional parameters
     * @returns {Object} {gifBlob, matches, inliers} or {error}
     */
    static async create(imgA, imgB, keypointsA, keypointsB, matches, options = {}) {
        const {
            maxSize = 800,
            frameDelay = 20, // 1/100th seconds (20 = 200ms)
            ransacThreshold = 5.0
        } = options;

        try {
            console.log(`[Wigglegram] Creating with ${matches.length} matches`);

            // Convert matches to point arrays
            const srcPts = matches.map(([idxA, idxB]) => [keypointsB[idxB].x, keypointsB[idxB].y]);
            const dstPts = matches.map(([idxA, idxB]) => [keypointsA[idxA].x, keypointsA[idxA].y]);

            if (srcPts.length < 4) {
                return { error: `Only ${srcPts.length} matches, need at least 4` };
            }

            // RANSAC homography
            console.log('[Wigglegram] Computing RANSAC homography...');
            const { H, inliers, inlierCount } = HomographyMatrixRANSAC.estimatePixels(srcPts, dstPts, { threshold: ransacThreshold });

            if (!H) {
                return { error: 'Failed to compute homography' };
            }

            console.log(`[Wigglegram] Found ${inlierCount} inliers`);

            // Get image data
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Draw image A
            canvas.width = imgA.width;
            canvas.height = imgA.height;
            ctx.drawImage(imgA, 0, 0);
            const imgDataA = ctx.getImageData(0, 0, imgA.width, imgA.height);

            // Draw image B
            canvas.width = imgB.width;
            canvas.height = imgB.height;
            ctx.drawImage(imgB, 0, 0);
            const imgDataB = ctx.getImageData(0, 0, imgB.width, imgB.height);

            // Warp image B to align with A
            console.log('[Wigglegram] Warping image B...');
            const warpedB = ImageWarp.perspectiveWarp(imgDataB, H, imgA.width, imgA.height);

            // Resize if needed
            let frameA = imgDataA;
            let frameB = warpedB;
            let outWidth = imgA.width;
            let outHeight = imgA.height;

            if (Math.max(outWidth, outHeight) > maxSize) {
                const scale = maxSize / Math.max(outWidth, outHeight);
                outWidth = Math.floor(outWidth * scale);
                outHeight = Math.floor(outHeight * scale);

                frameA = WigglegramGenerator.resizeImageData(imgDataA, outWidth, outHeight);
                frameB = WigglegramGenerator.resizeImageData(warpedB, outWidth, outHeight);
            }

            // Encode GIF
            console.log('[Wigglegram] Encoding GIF...');
            const gifData = GIFEncoder.encode([frameA, frameB], frameDelay);
            const gifBlob = new Blob([gifData], { type: 'image/gif' });

            console.log(`[Wigglegram] Created ${Math.round(gifBlob.size / 1024)}KB GIF`);

            return {
                gifBlob,
                matches: matches.length,
                inliers: inlierCount,
                // Also return the ImageData frames for WebGL rendering
                frameA,
                frameB,
                width: outWidth,
                height: outHeight,
                H // Return H for completeness if needed
            };

        } catch (error) {
            console.error('[Wigglegram] Error:', error);
            return { error: error.message };
        }
    }

    /**
     * Resize ImageData using bilinear interpolation
     */
    static resizeImageData(srcImg, dstWidth, dstHeight) {
        const srcData = srcImg.data;
        const srcWidth = srcImg.width;
        const srcHeight = srcImg.height;

        const dstData = new Uint8ClampedArray(dstWidth * dstHeight * 4);

        const xRatio = srcWidth / dstWidth;
        const yRatio = srcHeight / dstHeight;

        for (let dstY = 0; dstY < dstHeight; dstY++) {
            for (let dstX = 0; dstX < dstWidth; dstX++) {
                const srcX = dstX * xRatio;
                const srcY = dstY * yRatio;

                const x0 = Math.floor(srcX);
                const y0 = Math.floor(srcY);
                const x1 = Math.min(x0 + 1, srcWidth - 1);
                const y1 = Math.min(y0 + 1, srcHeight - 1);

                const fx = srcX - x0;
                const fy = srcY - y0;

                const w00 = (1 - fx) * (1 - fy);
                const w10 = fx * (1 - fy);
                const w01 = (1 - fx) * fy;
                const w11 = fx * fy;

                const idx00 = (y0 * srcWidth + x0) * 4;
                const idx10 = (y0 * srcWidth + x1) * 4;
                const idx01 = (y1 * srcWidth + x0) * 4;
                const idx11 = (y1 * srcWidth + x1) * 4;

                const dstIdx = (dstY * dstWidth + dstX) * 4;

                for (let c = 0; c < 4; c++) {
                    dstData[dstIdx + c] =
                        w00 * srcData[idx00 + c] +
                        w10 * srcData[idx10 + c] +
                        w01 * srcData[idx01 + c] +
                        w11 * srcData[idx11 + c];
                }
            }
        }

        return new ImageData(dstData, dstWidth, dstHeight);
    }
}

// Export for use as module
export { ImageWarp, GIFEncoder, WigglegramGenerator };
