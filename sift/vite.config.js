import { defineConfig } from 'vite';
import { resolve } from 'path';

// Keep this file as is for demos. I will create a separate config for library.
// Simple plugin to load wgsl as string
const wgslPlugin = () => {
    return {
        name: 'wgsl-loader',
        transform(code, id) {
            if (id.endsWith('.wgsl')) {
                return {
                    code: `export default ${JSON.stringify(code)};`,
                    map: null
                };
            }
        }
    };
};

export default defineConfig({
    plugins: [wgslPlugin()],
    root: './',
    publicDir: 'public',
    server: {
        port: 3000,
        open: true,
    },
    resolve: {
        alias: {
            '@': resolve(__dirname, './src'),
            'src': resolve(__dirname, './src'),
        },
    },
    build: {
        outDir: 'dist',
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                demo_webcam: resolve(__dirname, 'demos/demo_webcam.html'),
                demo_tracking: resolve(__dirname, 'demos/demo_webcam_tracking.html'),
                demo_wigglegram: resolve(__dirname, 'demos/demo_wigglegram.html'),
                demo_stereo: resolve(__dirname, 'demos/demo_stereo_pair.html'),
                unit_test: resolve(__dirname, 'tests/unit-test.html'),
                benchmark: resolve(__dirname, 'demos/benchmark.html'),
            },
        },
    },
    test: {
        include: ['tests/**/*.{test,spec}.{js,ts}'],
        browser: {
            enabled: true,
            name: 'chromium',
            provider: 'playwright',
            // headless: true // WebGPU might fail in headless without flags
        }
    },
});
