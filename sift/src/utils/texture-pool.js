/**
 * TexturePool
 * Manages a pool of reused textures to avoid allocation overhead.
 */
export class TexturePool {
    /**
     * @param {GPUDevice} device 
     * @param {Object} options 
     */
    constructor(device, options = {}) {
        this.device = device;
        this.pool = [];
        this.maxPoolSize = options.maxPoolSize || 10;
        this.lastUsed = 0;
    }

    /**
     * Acquires a texture of the specified dimensions and usage.
     * @param {number} width 
     * @param {number} height 
     * @param {string} format 
     * @param {number} usage 
     * @returns {GPUTexture}
     */
    acquire(width, height, format = 'rgba8unorm', usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT) {
        // Find a matching idle texture
        for (let i = 0; i < this.pool.length; i++) {
            const entry = this.pool[i];
            if (!entry.inUse && entry.width === width && entry.height === height && entry.format === format && entry.usage === usage) {
                entry.inUse = true;
                entry.lastUsed = Date.now();
                return entry.texture;
            }
        }

        // If pool is full, evict the oldest entry
        if (this.pool.length >= this.maxPoolSize) {
            this.evictOldest();
        }

        // Create new texture
        const texture = this.device.createTexture({
            size: [width, height],
            format: format,
            usage: usage
        });

        this.pool.push({
            texture,
            width,
            height,
            format,
            usage,
            inUse: true,
            lastUsed: Date.now()
        });

        return texture;
    }

    /**
     * Releases a texture back to the pool.
     * @param {GPUTexture} texture 
     */
    release(texture) {
        const entry = this.pool.find(e => e.texture === texture);
        if (entry) {
            entry.inUse = false;
        }
    }

    /**
     * Evicts the oldest unused texture from the pool.
     */
    evictOldest() {
        let oldestIdx = -1;
        let oldestTime = Infinity;

        for (let i = 0; i < this.pool.length; i++) {
            const entry = this.pool[i];
            if (!entry.inUse && entry.lastUsed < oldestTime) {
                oldestTime = entry.lastUsed;
                oldestIdx = i;
            }
        }

        if (oldestIdx !== -1) {
            this.pool[oldestIdx].texture.destroy();
            this.pool.splice(oldestIdx, 1);
        } else {
            // All are in use? Force destroy the first one if necessary, 
            // but usually we should just exceed the pool size slightly then.
            // For now, let's just allow it or destroy the first one.
            const first = this.pool.shift();
            first.texture.destroy();
        }
    }

    /**
     * Clears and destroys all textures in the pool.
     */
    destroy() {
        for (const entry of this.pool) {
            entry.texture.destroy();
        }
        this.pool = [];
    }
}
