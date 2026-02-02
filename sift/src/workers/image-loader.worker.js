/**
 * Image Loader Worker
 * Handles fetching and decoding images in a background thread.
 */
self.onmessage = async (e) => {
    const { id, source, options } = e.data;

    try {
        let blob;
        if (typeof source === 'string') {
            const res = await fetch(source);
            if (!res.ok) throw new Error(`Failed to fetch: ${source}`);
            blob = await res.blob();
        } else {
            blob = source;
        }

        // Decode to ImageBitmap
        // Options can include resizeWidth, resizeHeight, etc.
        const bitmap = await createImageBitmap(blob, options);

        // Transfer the bitmap back (zero-copy)
        self.postMessage({ id, bitmap }, [bitmap]);
    } catch (error) {
        self.postMessage({ id, error: error.message });
    }
};
