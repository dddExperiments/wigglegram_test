/**
 * Rotation utilities
 */
class Rotation {
    /**
     * Convert rotation matrix to axis-angle (Rodrigues) representation
     * @param {Array} R - 3x3 rotation matrix
     * @returns {Array} [rx, ry, rz] rotation vector (axis * angle)
     */
    static matrixToRodrigues(R) {
        const trace = R[0][0] + R[1][1] + R[2][2];
        const theta = Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2)));

        if (Math.abs(theta) < 1e-8) {
            return [0, 0, 0];
        }

        if (Math.abs(theta - Math.PI) < 1e-8) {
            // Handle 180 degree rotation
            const diag = [R[0][0], R[1][1], R[2][2]];
            const maxIdx = diag.indexOf(Math.max(...diag));
            const v = [R[0][maxIdx], R[1][maxIdx], R[2][maxIdx]];
            v[maxIdx] += 1;
            const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return v.map(x => x * Math.PI / norm);
        }

        const k = theta / (2 * Math.sin(theta));
        return [
            k * (R[2][1] - R[1][2]),
            k * (R[0][2] - R[2][0]),
            k * (R[1][0] - R[0][1])
        ];
    }

    /**
     * Convert axis-angle (Rodrigues) to rotation matrix
     * @param {Array} r - [rx, ry, rz] rotation vector
     * @returns {Array} 3x3 rotation matrix
     */
    static rodriguesToMatrix(r) {
        const theta = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

        if (theta < 1e-8) {
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
        }

        const k = [r[0] / theta, r[1] / theta, r[2] / theta];
        const c = Math.cos(theta);
        const s = Math.sin(theta);
        const v = 1 - c;

        return [
            [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
            [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
            [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v]
        ];
    }
}

export { Rotation };
