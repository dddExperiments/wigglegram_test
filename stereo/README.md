# LiteAnyStereo Web Demo

This is a WebGPU-accelerated web demo for LiteAnyStereo.

## Setup

1.  Ensure you have Node.js installed.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Copy the assets (if not already done):
    ```bash
    # (Windows)
    xcopy ..\assets public\assets /E /I /Y
    ```
4.  Copy the ONNX model (if updated):
    ```bash
    copy ..\LiteAnyStereo.onnx public\
    ```
5.  Copy ONNX Runtime WASM files (required for inference):
    ```bash
    copy node_modules\onnxruntime-web\dist\*.wasm public\
    ```

## Running

Start the development server:

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Notes

- **Initial Load**: The first run might take a moment to compile shaders.
- **WebGPU**: Requires a WebGPU-compatible browser (Chrome 113+).
- **Headers**: The server is configured to send Cross-Origin headers (`COOP`/`COEP`) to support SharedArrayBuffer, which is required for multi-threaded inference.
