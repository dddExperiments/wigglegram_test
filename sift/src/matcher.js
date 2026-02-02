/**
 * Base Matcher Class
 */
export class Matcher {
    constructor(options = {}) {
        this.options = {
            debug: false,
            ...options
        };
    }

    /**
     * Initialize the matcher (e.g. compile shaders)
     * @returns {Promise<void>}
     */
    async init() { }

    /**
     * Match descriptors
     * @param {Float32Array|Float32Array[]} descriptorsA 
     * @param {Float32Array|Float32Array[]} descriptorsB 
     * @param {number} ratio 
     * @returns {Promise<Array>} Matches [[idx1, idx2], ...]
     */
    async match(descriptorsA, descriptorsB, ratio = 0.75) {
        throw new Error('match() must be implemented');
    }

    log(...args) {
        if (this.options.debug) {
            console.log(...args);
        }
    }
}
