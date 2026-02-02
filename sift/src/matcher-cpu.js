import { Matcher } from './matcher.js';

export class MatcherCPU extends Matcher {
    constructor() {
        super();
    }

    /**
     * Brute-force matcher with Lowe's ratio test
     * @param {Float32Array[]} descriptors1 - Array of descriptors (usually 128 floats)
     * @param {Float32Array[]} descriptors2 - Array of descriptors
     * @param {number} ratio - Ratio threshold (default 0.75)
     * @returns {Promise<Array>} Array of matches [[idx1, idx2], ...]
     */
    async match(descriptors1, descriptors2, ratio = 0.75) {
        // Allow calling static logic if needed, but here we implement instance method
        return MatcherCPU.matchStatic(descriptors1, descriptors2, ratio);
    }

    /**
     * Static synchronous implementation (legacy support)
     */
    static matchStatic(descriptors1, descriptors2, ratio = 0.75) {
        const matches = [];
        const n1 = descriptors1.length;
        const n2 = descriptors2.length;

        // This is slow (O(N*M)), but fine for demo with < 1000 features
        for (let i = 0; i < n1; i++) {
            const d1 = descriptors1[i];
            let bestDist = Infinity;
            let secondBestDist = Infinity;
            let bestIdx = -1;

            for (let j = 0; j < n2; j++) {
                const d2 = descriptors2[j];

                // Euclidean distance squared
                let dist = 0;
                for (let k = 0; k < 128; k++) {
                    const diff = d1[k] - d2[k];
                    dist += diff * diff;
                }

                if (dist < bestDist) {
                    secondBestDist = bestDist;
                    bestDist = dist;
                    bestIdx = j;
                } else if (dist < secondBestDist) {
                    secondBestDist = dist;
                }
            }

            // Ratio test (using squared distances, so ratio^2)
            // if (d1/d2 < ratio) => d1^2 / d2^2 < ratio^2
            if (bestDist < ratio * ratio * secondBestDist) {
                matches.push([i, bestIdx]);
            }
        }

        return matches;
    }
}
