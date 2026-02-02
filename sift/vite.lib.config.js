import { defineConfig } from 'vite';
import { resolve } from 'path';

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
    build: {
        lib: {
            entry: resolve(__dirname, 'src/index.js'),
            name: 'WebSIFTGPU',
            fileName: (format) => `websiftgpu.${format}.js`,
        },
        rollupOptions: {
            // make sure to externalize deps that shouldn't be bundled
            external: ['three'],
            output: {
                globals: {
                    three: 'THREE',
                },
            },
        },
        outDir: 'dist/lib',
    },
});
