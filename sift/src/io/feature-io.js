/**
 * Feature I/O Utilities
 * Supports binary and ASCII (.sift) formats
 */

/**
 * Normalizes a SIFT descriptor for VisualSFM/Lowe's format.
 * L2 normalization -> Clipping to 0.2 -> Re-normalizing -> Scaling to [0, 255]
 * @param {Float32Array|Array} descriptor 
 * @returns {Uint8Array}
 */
export function normalizeDescriptor(descriptor) {
    const d = new Float32Array(descriptor);
    let norm = 0;
    for (let i = 0; i < d.length; i++) norm += d[i] * d[i];
    norm = Math.sqrt(norm);

    if (norm > 1e-6) {
        for (let i = 0; i < d.length; i++) d[i] /= norm;
    }

    // Clip to 0.2
    let normAfterClip = 0;
    for (let i = 0; i < d.length; i++) {
        d[i] = Math.min(0.2, d[i]);
        normAfterClip += d[i] * d[i];
    }
    normAfterClip = Math.sqrt(normAfterClip);

    // Re-normalize and scale to [0, 255]
    const result = new Uint8Array(d.length);
    if (normAfterClip > 1e-6) {
        for (let i = 0; i < d.length; i++) {
            // SIFT convention usually scales by 512 or similar to fill the byte range effectively
            // after the 0.2 clipping. 
            result[i] = Math.min(255, Math.floor((d[i] / normAfterClip) * 512));
        }
    }
    return result;
}

/**
 * Generates the content for a VisualSFM/Lowe .sift file (ASCII)
 * @param {Array} keypoints List of keypoint objects with .x, .y, .scale, .orientation, .descriptor
 * @returns {string}
 */
export function serializeVisualSFM(keypoints) {
    let content = `${keypoints.length} 128\n`;
    for (const kp of keypoints) {
        // Line format: y x scale orientation d0 d1 ... d127
        // Note: Lowe's format and VisualSFM often swap X/Y or use Y X. 
        // VisualSFM/COLMAP convention: X Y Scale Orientation
        // But the .sift file header-less part is often described as Y X.
        // We will follow the python script's convention which uses X Y.
        const normDesc = normalizeDescriptor(kp.descriptor || new Float32Array(128));
        const descStr = Array.from(normDesc).join(' ');
        content += `${kp.x.toFixed(4)} ${kp.y.toFixed(4)} ${kp.scale.toFixed(4)} ${kp.orientation.toFixed(4)} ${descStr}\n`;
    }
    return content;
}

/**
 * Triggers a browser download for the given content
 * @param {string|Blob} content 
 * @param {string} filename 
 */
export function downloadFile(content, filename) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Save features as VisualSFM ASCII .sift
 * @param {Array} keypoints 
 * @param {string} filename 
 */
export function downloadAsVisualSFM(keypoints, filename) {
    const content = serializeVisualSFM(keypoints);
    downloadFile(content, filename);
}

/**
 * Binary Format Save
 * Header (32 bytes) + Features
 * @param {Array} keypoints 
 * @param {number} width Original image width
 * @param {number} height Original image height
 * @returns {ArrayBuffer}
 */
export function serializeBinary(keypoints, width = 0, height = 0) {
    const HEADER_SIZE = 32;
    const FEATURE_SIZE = 16 + 512; // 4 floats (x,y,scale,ori) + 128 floats (descriptor)
    // Adding octave as 5th property (4 bytes)
    const FEATURE_SIZE_V2 = 20 + 512;

    const buffer = new ArrayBuffer(HEADER_SIZE + keypoints.length * FEATURE_SIZE_V2);
    const view = new DataView(buffer);

    // Header
    view.setUint8(0, 'W'.charCodeAt(0));
    view.setUint8(1, 'S'.charCodeAt(0));
    view.setUint8(2, 'F'.charCodeAt(0));
    view.setUint8(3, 'T'.charCodeAt(0));
    view.setUint32(4, 1, true); // Version
    view.setUint32(8, keypoints.length, true);
    view.setUint32(12, 128, true); // Dimension
    view.setUint32(16, width, true);
    view.setUint32(20, height, true);
    // 24-31 Reserved (already 0)

    let offset = HEADER_SIZE;
    for (const kp of keypoints) {
        view.setFloat32(offset + 0, kp.x, true);
        view.setFloat32(offset + 4, kp.y, true);
        view.setFloat32(offset + 8, kp.scale, true);
        view.setFloat32(offset + 12, kp.orientation, true);
        view.setInt32(offset + 16, kp.octave || 0, true);

        const desc = kp.descriptor || new Float32Array(128);
        for (let i = 0; i < 128; i++) {
            view.setFloat32(offset + 20 + i * 4, desc[i], true);
        }
        offset += FEATURE_SIZE_V2;
    }

    return buffer;
}

/**
 * Download features as binary file
 * @param {Array} keypoints 
 * @param {string} filename 
 * @param {number} width 
 * @param {number} height 
 */
export function downloadAsBinary(keypoints, filename, width = 0, height = 0) {
    const buffer = serializeBinary(keypoints, width, height);
    downloadFile(new Blob([buffer], { type: 'application/octet-stream' }), filename);
}

/**
 * Loading Utilities
 */

/**
 * Loads features from a binary ArrayBuffer
 * @param {ArrayBuffer} buffer 
 * @returns {Object} { keypoints, width, height }
 */
export function loadBinary(buffer) {
    const HEADER_SIZE = 32;
    const view = new DataView(buffer);

    // Check magic
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== "WSFT") throw new Error("Invalid binary format");

    const version = view.getUint32(4, true);
    const count = view.getUint32(8, true);
    const dim = view.getUint32(12, true);
    const width = view.getUint32(16, true);
    const height = view.getUint32(20, true);

    const keypoints = [];
    let offset = HEADER_SIZE;

    // Version 1 size: 20 (x,y,scale,ori,octave) + dim*4
    const featureStride = 20 + dim * 4;

    for (let i = 0; i < count; i++) {
        const kp = {
            x: view.getFloat32(offset + 0, true),
            y: view.getFloat32(offset + 4, true),
            scale: view.getFloat32(offset + 8, true),
            orientation: view.getFloat32(offset + 12, true),
            octave: view.getInt32(offset + 16, true),
            descriptor: new Float32Array(dim)
        };

        for (let j = 0; j < dim; j++) {
            kp.descriptor[j] = view.getFloat32(offset + 20 + j * 4, true);
        }

        keypoints.push(kp);
        offset += featureStride;
    }

    return { keypoints, width, height, version };
}

/**
 * Loads features from a VisualSFM ASCII string
 * @param {string} content 
 * @returns {Object} { keypoints }
 */
export function loadVisualSFM(content) {
    const lines = content.split('\n');
    if (lines.length === 0) return { keypoints: [] };

    const header = lines[0].trim().split(/\s+/);
    if (header.length < 2) return { keypoints: [] };

    const count = parseInt(header[0]);
    const dim = parseInt(header[1]);

    const keypoints = [];
    for (let i = 1; i <= count; i++) {
        const line = lines[i]?.trim();
        if (!line) continue;

        const parts = line.split(/\s+/);
        if (parts.length < 4 + dim) continue;

        const kp = {
            x: parseFloat(parts[0]),
            y: parseFloat(parts[1]),
            scale: parseFloat(parts[2]),
            orientation: parseFloat(parts[3]),
            descriptor: new Float32Array(dim)
        };

        // Convert from byte [0, 255] back to normalized float
        for (let j = 0; j < dim; j++) {
            kp.descriptor[j] = parseFloat(parts[4 + j]) / 128.0; // Approximation (if it was scaled by 512 and then clipped, this is tricky)
            // Actually, if we use the same 512 scaling, we should use 512 here.
            // But let's just re-normalize anyway.
        }

        // Re-normalize loaded descriptor
        let norm = 0;
        for (let j = 0; j < dim; j++) norm += kp.descriptor[j] * kp.descriptor[j];
        norm = Math.sqrt(norm);
        if (norm > 1e-6) {
            for (let j = 0; j < dim; j++) kp.descriptor[j] /= norm;
        }

        keypoints.push(kp);
    }

    return { keypoints };
}

/**
 * Loads any supported feature file
 * @param {File|Blob} file 
 * @returns {Promise<Object>}
 */
export async function loadFeatures(file) {
    if (file.name && (file.name.endsWith('.sift') || file.name.endsWith('.txt'))) {
        const text = await file.text();
        return loadVisualSFM(text);
    } else {
        const buffer = await file.arrayBuffer();
        try {
            return loadBinary(buffer);
        } catch (e) {
            // Fallback to text if binary load fails
            const text = new TextDecoder().decode(buffer);
            return loadVisualSFM(text);
        }
    }
}

