import { describe, it, expect } from 'vitest';
import { SIFTCPU } from '../src/sift-cpu.js';

describe('SIFTCPU', () => {
    it('should be instantiable', () => {
        const cpu = new SIFTCPU();
        expect(cpu).toBeDefined();
        // SIFTCPU usually has width/height initialized to 0 or null
        expect(cpu.currentImage).toBeNull();
    });

    // We can add more tests here, e.g. math utils if exported
});
