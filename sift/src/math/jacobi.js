/**
 * Robust Jacobi Eigenvalue Algorithm for Symmetric Matrices
 */
class JacobiEigenvalue {
    static compute(A, maxIter = 100) {
        const n = A.length;
        // V starts as identity
        const V = [];
        for (let i = 0; i < n; i++) {
            V[i] = new Array(n).fill(0);
            V[i][i] = 1.0;
        }

        // Copy A to D (diagonal will be eigenvalues)
        let D = A.map(row => [...row]);

        for (let iter = 0; iter < maxIter; iter++) {
            // Find max off-diagonal element
            let maxVal = 0;
            let p = 0, q = 0;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(D[i][j]) > maxVal) {
                        maxVal = Math.abs(D[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxVal < 1e-12) break; // Converged

            // Calculate rotation
            const app = D[p][p];
            const aqq = D[q][q];
            const apq = D[p][q];

            const tau = (aqq - app) / (2 * apq);
            const t = (tau >= 0) ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
            const c = 1 / Math.sqrt(1 + t * t);
            const s = t * c;

            // Rotate D
            D[p][p] -= t * apq;
            D[q][q] += t * apq;
            D[p][q] = 0;
            D[q][p] = 0;

            for (let i = 0; i < n; i++) {
                if (i !== p && i !== q) {
                    const dip = D[i][p];
                    const diq = D[i][q];
                    D[i][p] = c * dip - s * diq;
                    D[p][i] = D[i][p];
                    D[i][q] = s * dip + c * diq;
                    D[q][i] = D[i][q];
                }
            }

            // Accumulate V
            // V = V * R. Columns are eigenvectors.
            // V[i][p] = c * V[i][p] - s * V[i][q]
            for (let i = 0; i < n; i++) {
                const vip = V[i][p];
                const viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }

        // Extract eigenvalues and eigenvectors
        const eigenvalues = [];
        const eigenvectors = [];

        for (let i = 0; i < n; i++) {
            eigenvalues.push(D[i][i]);
            // Get i-th column of V as eigenvector
            const vec = [];
            for (let j = 0; j < n; j++) vec.push(V[j][i]);
            eigenvectors.push(vec);
        }

        return { eigenvalues, eigenvectors };
    }
}

export { JacobiEigenvalue };
