/**
 * Homography Matrix Estimation
 */
import { Matrix } from '../math/matrix.js';
import { JacobiEigenvalue } from '../math/jacobi.js';
import { hartleyNormalize } from './normalization.js';
import { CameraIntrinsics } from './camera_intrinsics.js'; // Imported for types

class HomographyMatrix {
    /**
     * Compute Homography H such that x2 = H * x1
     * using normalized DLT (4-point algorithm)
     * @param {Array} pts1 - Points from image 1 [[x,y], ...] (at least 4)
     * @param {Array} pts2 - Points from image 2 [[x,y], ...]
     * @returns {Array|null} 3x3 Homography Matrix or null if failed
     */
    static compute4Point(pts1, pts2) {
        if (pts1.length < 4) return null;

        const { normalized: norm1, T: T1 } = hartleyNormalize(pts1);
        const { normalized: norm2, T: T2 } = hartleyNormalize(pts2);

        // Build constraint matrix A (2n x 9)
        const A = [];
        for (let i = 0; i < pts1.length; i++) {
            const [x, y] = norm1[i];
            const [xp, yp] = norm2[i];

            // Ax = 0
            // [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            // [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
            A.push([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]);
            A.push([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]);
        }

        // Solve using SVD (A^T * A)
        const At = Matrix.transpose(A);
        const AtA = Matrix.multiply(At, A);
        const result = JacobiEigenvalue.compute(AtA);
        if (!result) return null;

        // Smallest eigenvalue index
        let minIdx = 0;
        let minVal = Math.abs(result.eigenvalues[0]);
        for (let i = 1; i < result.eigenvalues.length; i++) {
            if (Math.abs(result.eigenvalues[i]) < minVal) {
                minVal = Math.abs(result.eigenvalues[i]);
                minIdx = i;
            }
        }
        const h = result.eigenvectors[minIdx];

        // Reshape to 3x3
        let H = [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], h[8]]
        ];

        // Denormalize: H = T2^-1 * H_norm * T1
        // T2 is [scale, 0, tx; 0, scale, ty; 0, 0, 1]
        // T2^-1 is [1/scale, 0, -tx/scale; ...]
        const T2inv = [
            [1 / T2[0][0], 0, -T2[0][2] / T2[0][0]],
            [0, 1 / T2[1][1], -T2[1][2] / T2[1][1]],
            [0, 0, 1]
        ];

        H = Matrix.multiply(Matrix.multiply(T2inv, H), T1);

        // Normalize H so H[2][2] = 1 (if not 0)
        if (Math.abs(H[2][2]) > 1e-8) {
            const s = 1 / H[2][2];
            for (let i = 0; i < 3; i++)
                for (let j = 0; j < 3; j++) H[i][j] *= s;
        }

        return H;
    }

    /**
     * Compute symmetric transfer error for H
     * d(x2, H*x1)^2 + d(x1, H^-1*x2)^2
     * @param {Array} H - 3x3 Homography Matrix
     * @param {Array} pt1 - [x, y] normalized
     * @param {Array} pt2 - [x, y] normalized
     * @returns {number} Error squared
     */
    static symmetricTransferError(H, pt1, pt2) {
        const [x1, y1] = pt1;
        const [x2, y2] = pt2;

        // Forward: x2_est = H * x1
        const w1 = H[2][0] * x1 + H[2][1] * y1 + H[2][2];
        const x2_est = (H[0][0] * x1 + H[0][1] * y1 + H[0][2]) / w1;
        const y2_est = (H[1][0] * x1 + H[1][1] * y1 + H[1][2]) / w1;

        const err1 = (x2 - x2_est) ** 2 + (y2 - y2_est) ** 2;

        // Backward: x1_est = H^-1 * x2
        // Invert H (3x3)
        const det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) -
            H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
            H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);

        if (Math.abs(det) < 1e-10) return Infinity;

        const invDet = 1 / det;
        const Hinv = [
            [(H[1][1] * H[2][2] - H[1][2] * H[2][1]) * invDet, (H[0][2] * H[2][1] - H[0][1] * H[2][2]) * invDet, (H[0][1] * H[1][2] - H[0][2] * H[1][1]) * invDet],
            [(H[1][2] * H[2][0] - H[1][0] * H[2][2]) * invDet, (H[0][0] * H[2][2] - H[0][2] * H[2][0]) * invDet, (H[0][2] * H[1][0] - H[0][0] * H[1][2]) * invDet],
            [(H[1][0] * H[2][1] - H[1][1] * H[2][0]) * invDet, (H[0][1] * H[2][0] - H[0][0] * H[2][1]) * invDet, (H[0][0] * H[1][1] - H[0][1] * H[1][0]) * invDet]
        ];

        const w2 = Hinv[2][0] * x2 + Hinv[2][1] * y2 + Hinv[2][2];
        const x1_est = (Hinv[0][0] * x2 + Hinv[0][1] * y2 + Hinv[0][2]) / w2;
        const y1_est = (Hinv[1][0] * x2 + Hinv[1][1] * y2 + Hinv[1][2]) / w2;

        const err2 = (x1 - x1_est) ** 2 + (y1 - y1_est) ** 2;

        return err1 + err2;
    }
}

/**
 * RANSAC for Homography Matrix estimation
 */
class HomographyMatrixRANSAC {
    /**
     * Estimate H with RANSAC
     * @param {Array} pts1 - Points from image 1 [[x,y], ...] in PIXEL coords
     * @param {Array} pts2 - Points from image 2 [[x,y], ...] in PIXEL coords
     * @param {CameraIntrinsics} K1 - Intrinsics for image 1
     * @param {CameraIntrinsics} K2 - Intrinsics for image 2
     * @param {Object} options - { threshold, maxIterations, confidence }
     * @returns {Object} { H, inliers, inlierCount, inlierRatio }
     */
    static estimate(pts1, pts2, K1, K2, options = {}) {
        const {
            threshold = 5.0, // pixels (stricter for H usually)
            maxIterations = 1000,
            confidence = 0.99
        } = options;

        const n = pts1.length;
        if (n < 4) {
            return { H: null, inliers: [], inlierCount: 0, inlierRatio: 0 };
        }

        // Normalize coordinates
        const norm1 = pts1.map(([x, y]) => K1.normalize(x, y));
        const norm2 = pts2.map(([x, y]) => K2.normalize(x, y));

        // Threshold in normalized coords? 
        // No, transfer error is usually geometric distance.
        // But our inputs are pixels.
        // Ideally we compute H on normalized points, then map error back to pixels?
        // Or just compute on normalized and convert threshold.
        // Let's compute H on normalized points.
        const normThreshold = threshold / Math.max(K1.fx, K1.fy);
        const thresholdSq = normThreshold * normThreshold;

        let bestH = null;
        let bestInliers = [];
        let bestInlierCount = 0;
        let iterations = maxIterations;

        for (let iter = 0; iter < iterations; iter++) {
            const indices = [];
            while (indices.length < 4) {
                const idx = Math.floor(Math.random() * n);
                if (!indices.includes(idx)) indices.push(idx);
            }

            const samplePts1 = indices.map(i => norm1[i]);
            const samplePts2 = indices.map(i => norm2[i]);

            const H = HomographyMatrix.compute4Point(samplePts1, samplePts2);
            if (!H) continue;

            const inliers = [];
            for (let i = 0; i < n; i++) {
                // Use forward error only for speed in RANSAC loop
                const [x1, y1] = norm1[i];
                const [x2, y2] = norm2[i];

                const w = H[2][0] * x1 + H[2][1] * y1 + H[2][2];
                if (Math.abs(w) < 1e-10) continue;

                const x2_est = (H[0][0] * x1 + H[0][1] * y1 + H[0][2]) / w;
                const y2_est = (H[1][0] * x1 + H[1][1] * y1 + H[1][2]) / w;

                const errSq = (x2 - x2_est) ** 2 + (y2 - y2_est) ** 2;

                if (errSq < thresholdSq) {
                    inliers.push(i);
                }
            }

            if (inliers.length > bestInlierCount) {
                bestInlierCount = inliers.length;
                bestInliers = inliers;
                bestH = H;

                const inlierRatio = bestInlierCount / n;
                if (inlierRatio > 0.1) {
                    const p = confidence;
                    const newIterations = Math.ceil(
                        Math.log(1 - p) / Math.log(1 - Math.pow(inlierRatio, 4))
                    );
                    iterations = Math.min(maxIterations, Math.max(iter + 1, newIterations));
                }
            }
        }

        return {
            H: bestH,
            inliers: bestInliers,
            inlierCount: bestInlierCount,
            inlierRatio: bestInlierCount / n
        };
    }
    static estimatePixels(pts1, pts2, options = {}) {
        const {
            threshold = 5.0,
            maxIterations = 1000,
            confidence = 0.99
        } = options;

        const n = pts1.length;
        if (n < 4) {
            return { H: null, inliers: [], inlierCount: 0, inlierRatio: 0 };
        }

        const thresholdSq = threshold * threshold;

        let bestH = null;
        let bestInliers = [];
        let bestInlierCount = 0;
        let iterations = maxIterations;

        for (let iter = 0; iter < iterations; iter++) {
            const indices = [];
            while (indices.length < 4) {
                const idx = Math.floor(Math.random() * n);
                if (!indices.includes(idx)) indices.push(idx);
            }

            const samplePts1 = indices.map(i => pts1[i]);
            const samplePts2 = indices.map(i => pts2[i]);

            const H = HomographyMatrix.compute4Point(samplePts1, samplePts2);
            if (!H) continue;

            const inliers = [];
            for (let i = 0; i < n; i++) {
                // Use forward error only for speed
                const [x1, y1] = pts1[i];
                const [x2, y2] = pts2[i];

                const w = H[2][0] * x1 + H[2][1] * y1 + H[2][2];
                if (Math.abs(w) < 1e-10) continue;

                const x2_est = (H[0][0] * x1 + H[0][1] * y1 + H[0][2]) / w;
                const y2_est = (H[1][0] * x1 + H[1][1] * y1 + H[1][2]) / w;

                const errSq = (x2 - x2_est) ** 2 + (y2 - y2_est) ** 2;

                if (errSq < thresholdSq) {
                    inliers.push(i);
                }
            }

            if (inliers.length > bestInlierCount) {
                bestInlierCount = inliers.length;
                bestInliers = inliers;
                bestH = H;

                const inlierRatio = bestInlierCount / n;
                if (inlierRatio > 0.1) {
                    const p = confidence;
                    const newIterations = Math.ceil(
                        Math.log(1 - p) / Math.log(1 - Math.pow(inlierRatio, 4))
                    );
                    iterations = Math.min(maxIterations, Math.max(iter + 1, newIterations));
                }
            }
        }

        return {
            H: bestH,
            inliers: bestInliers,
            inlierCount: bestInlierCount,
            inlierRatio: bestInlierCount / n
        };
    }
}

export { HomographyMatrix, HomographyMatrixRANSAC };
