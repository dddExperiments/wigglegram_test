/**
 * Matrix Math Utilities
 */

import { JacobiEigenvalue } from './jacobi.js';

class Matrix {
    /**
     * Enforce rank-2 constraint on 3x3 matrix
     * Defines M' = U * diag(s1, s2, 0) * V^T
     */
    static enforceRank2(M) {
        // Compute M * M^T
        const Mt = Matrix.transpose(M);
        const MMt = Matrix.multiply(M, Mt);

        // Get eigenvectors of MMt (U)
        const resU = JacobiEigenvalue.compute(MMt);
        const indicesU = resU.eigenvalues.map((v, i) => ({ v: Math.abs(v), i }))
            .sort((a, b) => b.v - a.v)
            .map(o => o.i);
        const u1 = resU.eigenvectors[indicesU[0]];
        const u2 = resU.eigenvectors[indicesU[1]];
        // Ensure u3 is cross product for RH system consistency, though strict SVD might not care.
        // But Essential Matrix logic relied on it.
        const u3 = Matrix.cross(u1, u2);

        // Compute s1, s2
        // y1 = MMt * u1
        const y1 = MMt.map(row => row.reduce((s, v, j) => s + v * u1[j], 0));
        const s1sq = y1.reduce((s, v, i) => s + v * u1[i], 0);
        const s1 = Math.sqrt(Math.max(0, s1sq));

        const y2 = MMt.map(row => row.reduce((s, v, j) => s + v * u2[j], 0));
        const s2sq = y2.reduce((s, v, i) => s + v * u2[i], 0);
        const s2 = Math.sqrt(Math.max(0, s2sq));

        // Get eigenvectors of MtM (V)
        // M = U S V^T  => M^T M = V S U^T U S V^T = V S^2 V^T
        const MtM = Matrix.multiply(Mt, M);
        const resV = JacobiEigenvalue.compute(MtM);
        const indicesV = resV.eigenvalues.map((v, i) => ({ v: Math.abs(v), i }))
            .sort((a, b) => b.v - a.v)
            .map(o => o.i);

        const v1 = resV.eigenvectors[indicesV[0]];
        const v2 = resV.eigenvectors[indicesV[1]];
        // We need to ensuring consistent signs between U and V so that M = U S V^T holds.
        // With U and S fixed, we check sign of V.
        // u_i = M * v_i / s_i

        const u1_check = [0, 0, 0];
        if (s1 > 1e-8) {
            u1_check[0] = (M[0][0] * v1[0] + M[0][1] * v1[1] + M[0][2] * v1[2]) / s1;
            u1_check[1] = (M[1][0] * v1[0] + M[1][1] * v1[1] + M[1][2] * v1[2]) / s1;
            u1_check[2] = (M[2][0] * v1[0] + M[2][1] * v1[1] + M[2][2] * v1[2]) / s1;
            // Dot product to check orientation
            const dot = u1_check[0] * u1[0] + u1_check[1] * u1[1] + u1_check[2] * u1[2];
            if (dot < 0) {
                // Flip v1
                v1[0] = -v1[0]; v1[1] = -v1[1]; v1[2] = -v1[2];
            }
        }

        const u2_check = [0, 0, 0];
        if (s2 > 1e-8) {
            u2_check[0] = (M[0][0] * v2[0] + M[0][1] * v2[1] + M[0][2] * v2[2]) / s2;
            u2_check[1] = (M[1][0] * v2[0] + M[1][1] * v2[1] + M[1][2] * v2[2]) / s2;
            u2_check[2] = (M[2][0] * v2[0] + M[2][1] * v2[1] + M[2][2] * v2[2]) / s2;
            const dot = u2_check[0] * u2[0] + u2_check[1] * u2[1] + u2_check[2] * u2[2];
            if (dot < 0) {
                // Flip v2
                v2[0] = -v2[0]; v2[1] = -v2[1]; v2[2] = -v2[2];
            }
        }

        // Reconstruct M with singular values (s1, s2, 0)
        // M' = s1 * u1 * v1^T + s2 * u2 * v2^T
        const M_new = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                M_new[r][c] = s1 * u1[r] * v1[c] + s2 * u2[r] * v2[c];
            }
        }

        return M_new;
    }

    /**
     * Cross product of two 3D vectors
     */
    static cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    /**
     * Multiply two matrices (A * B)
     */
    static multiply(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;
        const result = [];

        for (let i = 0; i < rowsA; i++) {
            result[i] = [];
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * Transpose a matrix
     */
    static transpose(A) {
        const rows = A.length;
        const cols = A[0].length;
        const result = [];

        for (let j = 0; j < cols; j++) {
            result[j] = [];
            for (let i = 0; i < rows; i++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }

    /**
     * Invert a 3x3 matrix
     */
    static invert3x3(M) {
        const [[a, b, c], [d, e, f], [g, h, i]] = M;

        const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

        if (Math.abs(det) < 1e-10) {
            return null; // Singular matrix
        }

        const invDet = 1 / det;

        return [
            [(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
            [(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
            [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet]
        ];
    }

    /**
     * Solve Ax = b using least squares (via normal equations)
     * Returns x that minimizes ||Ax - b||^2
     */
    static solveLeastSquares(A, b) {
        const At = Matrix.transpose(A);
        const AtA = Matrix.multiply(At, A);
        const Atb = Matrix.multiply(At, b.map(v => [v]));

        // For 2x2 AtA, solve directly
        if (AtA.length === 2) {
            const det = AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0];
            if (Math.abs(det) < 1e-10) return null;

            return [
                (AtA[1][1] * Atb[0][0] - AtA[0][1] * Atb[1][0]) / det,
                (-AtA[1][0] * Atb[0][0] + AtA[0][0] * Atb[1][0]) / det
            ];
        }

        // For larger systems, use Gaussian elimination
        return Matrix.solveGaussian(AtA, Atb.map(v => v[0]));
    }

    /**
     * Gaussian elimination with partial pivoting
     */
    static solveGaussian(A, b) {
        const n = A.length;

        // Create augmented matrix
        const aug = A.map((row, i) => [...row, b[i]]);

        // Forward elimination
        for (let col = 0; col < n; col++) {
            // Find pivot
            let maxRow = col;
            for (let row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
                    maxRow = row;
                }
            }
            [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

            if (Math.abs(aug[col][col]) < 1e-10) {
                return null; // Singular
            }

            // Eliminate below
            for (let row = col + 1; row < n; row++) {
                const factor = aug[row][col] / aug[col][col];
                for (let j = col; j <= n; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        const x = new Array(n);
        for (let i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (let j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        return x;
    }
}

export { Matrix };
