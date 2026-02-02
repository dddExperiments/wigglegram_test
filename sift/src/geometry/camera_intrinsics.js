/**
 * Camera intrinsics (simple pinhole model)
 */
class CameraIntrinsics {
    /**
     * @param {number} fx - Focal length in pixels (x)
     * @param {number} fy - Focal length in pixels (y)
     * @param {number} cx - Principal point x
     * @param {number} cy - Principal point y
     */
    /**
     * @param {number} fx - Focal length in pixels (x)
     * @param {number} fy - Focal length in pixels (y)
     * @param {number} cx - Principal point x
     * @param {number} cy - Principal point y
     * @param {number} k1 - Radial distortion coeff 1 (default 0)
     * @param {number} k2 - Radial distortion coeff 2 (default 0)
     */
    constructor(fx, fy, cx, cy, k1 = 0, k2 = 0) {
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
        this.k1 = k1;
        this.k2 = k2;

        // 3x3 intrinsic matrix K
        this.K = [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ];

        // Inverse of K
        this.Kinv = [
            [1 / fx, 0, -cx / fx],
            [0, 1 / fy, -cy / fy],
            [0, 0, 1]
        ];
    }

    /**
     * Create intrinsics from EXIF data
     * Uses FocalPlaneResolution to compute accurate sensor size when available
     * 
     * @param {Object} exifInfo - Parsed EXIF info
     * @param {number} imageWidth - Image width in pixels
     * @param {number} imageHeight - Image height in pixels
     * @returns {CameraIntrinsics}
     */
    static fromExif(exifInfo, imageWidth, imageHeight) {
        let focalPixels;

        const exif = exifInfo?.tags?.Exif;
        const focalLengthMM = exif?.FocalLength?.value;

        // Method 1: Use FocalPlaneResolution (most accurate - what VisualSFM uses)
        // FocalPlaneXResolution gives pixels per unit (inch/cm) of the sensor
        const fpXRes = exif?.FocalPlaneXResolution?.value;
        const fpYRes = exif?.FocalPlaneYResolution?.value;
        const fpUnit = exif?.FocalPlaneResolutionUnit?.value; // 2=inch, 3=cm

        if (focalLengthMM && fpXRes) {
            // Convert FocalPlaneResolution to pixels per mm
            let pixelsPerMM;
            if (fpUnit === 3) {
                // Unit is centimeters
                pixelsPerMM = fpXRes / 10;
            } else {
                // Default to inches (fpUnit === 2 or undefined)
                pixelsPerMM = fpXRes / 25.4;
            }

            // Focal length in pixels at ORIGINAL sensor resolution
            const focalPixelsOriginal = focalLengthMM * pixelsPerMM;

            // Get original sensor width from EXIF (PixelXDimension or ExifImageWidth)
            const originalWidth = exif?.PixelXDimension?.value ||
                exif?.ExifImageWidth?.value ||
                exifInfo?.tags?.Tiff?.ImageWidth?.value;

            if (originalWidth && originalWidth !== imageWidth) {
                // Image has been resized - scale focal length proportionally
                const scaleFactor = imageWidth / originalWidth;
                focalPixels = focalPixelsOriginal * scaleFactor;

                console.log(`[Intrinsics] Using FocalPlane: f=${focalLengthMM}mm, original=${originalWidth}px, resized=${imageWidth}px, scale=${scaleFactor.toFixed(3)} => ${focalPixels.toFixed(1)} px`);
            } else {
                // No resize info or same size
                focalPixels = focalPixelsOriginal;
                console.log(`[Intrinsics] Using FocalPlane: f=${focalLengthMM}mm, ${pixelsPerMM.toFixed(1)} px/mm => ${focalPixels.toFixed(1)} px`);
            }
        }
        // Method 2: Use FocalLengthIn35mmFilm equivalent
        else if (exif?.FocalLengthIn35mmFilm?.value) {
            const f35 = exif.FocalLengthIn35mmFilm.value;
            // 35mm film has 36mm width, compute ratio to displayed image
            const sensorWidthMM = 36; // Full frame
            focalPixels = (f35 / sensorWidthMM) * Math.max(imageWidth, imageHeight);

            console.log(`[Intrinsics] Using 35mm equiv: f35=${f35}mm => ${focalPixels.toFixed(1)} px`);
        }
        // Method 3: Use FocalLength with estimated sensor size
        else if (focalLengthMM) {
            // Estimate sensor size from image aspect ratio
            // Common sensor sizes: APS-C (23.5 x 15.6mm), Full frame (36 x 24mm)
            // Use a heuristic based on typical compact/phone cameras (~6mm sensor width for 4:3)
            const aspectRatio = imageWidth / imageHeight;
            let sensorWidthMM;

            if (aspectRatio > 1.4) {
                // 3:2 or wider - likely DSLR/mirrorless
                sensorWidthMM = 23.5; // APS-C default
            } else {
                // 4:3 - likely compact or phone
                sensorWidthMM = 6.17; // 1/2.3" sensor common in compacts
            }

            focalPixels = (focalLengthMM / sensorWidthMM) * imageWidth;

            console.log(`[Intrinsics] Estimated sensor ${sensorWidthMM}mm: f=${focalLengthMM}mm => ${focalPixels.toFixed(1)} px`);
        }
        // Method 4: Last resort - assume 50mm equivalent on full-frame
        else {
            const f35Equiv = 50;
            focalPixels = (f35Equiv / 36) * Math.max(imageWidth, imageHeight);

            console.log(`[Intrinsics] No EXIF focal: assuming 50mm equiv => ${focalPixels.toFixed(1)} px`);
        }

        // Principal point at image center
        const cx = imageWidth / 2;
        const cy = imageHeight / 2;

        return new CameraIntrinsics(focalPixels, focalPixels, cx, cy);
    }

    /**
     * Normalize a point from pixel to normalized image coordinates
     * Note: This implements the LINEAR model. For accurate results with distortion, separate undistortion is needed.
     * @param {number} x - Pixel x
     * @param {number} y - Pixel y
     * @returns {Array} [xn, yn] normalized coordinates
     */
    normalize(x, y) {
        return [
            (x - this.cx) / this.fx,
            (y - this.cy) / this.fy
        ];
    }

    /**
     * Denormalize a point from normalized to pixel coordinates
     * @param {number} xn - Normalized x
     * @param {number} yn - Normalized y
     * @returns {Array} [x, y] pixel coordinates
     */
    denormalize(xn, yn) {
        return [
            xn * this.fx + this.cx,
            yn * this.fy + this.cy
        ];
    }
}

export { CameraIntrinsics };
