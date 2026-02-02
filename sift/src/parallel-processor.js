/**
 * ParallelImageLoader
 * Manages a pool of workers for concurrent image decoding.
 */
export class ParallelImageLoader {
    /**
     * @param {number} workerCount Number of worker threads to spawn
     */
    constructor(workerCount = 4) {
        this.workerCount = workerCount;
        this.workers = [];
        this.idleWorkers = [];
        this.queue = [];
        this.pendingTasks = new Map();
        this.taskIdCounter = 0;

        this.init();
    }

    init() {
        const workerUrl = new URL('./workers/image-loader.worker.js', import.meta.url).href;
        for (let i = 0; i < this.workerCount; i++) {
            const worker = new Worker(workerUrl, { type: 'module' });
            worker.onmessage = (e) => this.handleWorkerMessage(worker, e.data);
            this.workers.push(worker);
            this.idleWorkers.push(worker);
        }
    }

    /**
     * Adds an image to the loading queue.
     * @param {string|Blob} source 
     * @param {Object} options createImageBitmap options
     * @returns {Promise<ImageBitmap>}
     */
    async load(source, options = {}) {
        return new Promise((resolve, reject) => {
            const id = this.taskIdCounter++;
            this.pendingTasks.set(id, { resolve, reject });
            this.queue.push({ id, source, options });
            this.processQueue();
        });
    }

    processQueue() {
        while (this.idleWorkers.length > 0 && this.queue.length > 0) {
            const worker = this.idleWorkers.pop();
            const task = this.queue.shift();
            worker.postMessage(task);
        }
    }

    handleWorkerMessage(worker, data) {
        const { id, bitmap, error } = data;
        const task = this.pendingTasks.get(id);

        if (task) {
            this.pendingTasks.delete(id);
            if (error) task.reject(new Error(error));
            else task.resolve(bitmap);
        }

        this.idleWorkers.push(worker);
        this.processQueue();
    }

    /**
     * Loads multiple images in parallel.
     * @param {Array<string|Blob>} sources 
     * @param {Object} options 
     * @returns {Promise<Array<ImageBitmap>>}
     */
    async loadAll(sources, options = {}) {
        return Promise.all(sources.map(s => this.load(s, options)));
    }

    /**
     * Terminates all workers.
     */
    destroy() {
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
        this.idleWorkers = [];
        this.pendingTasks.clear();
        this.queue = [];
    }
}
