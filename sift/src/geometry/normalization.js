/**
 * Geometric Normalization Utilities
 */

/**
 * Hartley normalization: translate centroid to origin, scale so avg distance = sqrt(2)
 * @param {Array} pts - Points [[x,y], ...]
 * @returns {Object} { normalized, T }
 */
export function hartleyNormalize(pts) {
    const n = pts.length;

    // Compute centroid
    let cx = 0, cy = 0;
    for (const [x, y] of pts) {
        cx += x;
        cy += y;
    }
    cx /= n;
    cy /= n;

    // Compute average distance from centroid
    let avgDist = 0;
    for (const [x, y] of pts) {
        avgDist += Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
    }
    avgDist /= n;

    // Scale factor: sqrt(2) / avgDist
    const scale = Math.sqrt(2) / avgDist;

    // Transformation matrix T
    const T = [
        [scale, 0, -scale * cx],
        [0, scale, -scale * cy],
        [0, 0, 1]
    ];

    // Normalize points
    const normalized = [];
    for (const [x, y] of pts) {
        normalized.push([
            (x - cx) * scale,
            (y - cy) * scale
        ]);
    }

    return { normalized, T };
}
