
import * as ort from 'onnxruntime-web';
import { PointCloudRenderer } from './src/renderer.js';

// Initialize ORT
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency / 2);
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = "./";

const MODEL_PATH = './LiteAnyStereo.onnx';

// UI Elements
const sampleSelect = document.getElementById('sample-select');
const runBtn = document.getElementById('run-btn');
const statusBar = document.getElementById('status-bar');
const imgLeft = document.getElementById('img-left');
const imgRight = document.getElementById('img-right');
const canvasResult = document.getElementById('canvas-result');
const canvas3d = document.getElementById('canvas-3d');
const loadingOverlay = document.getElementById('loading-overlay');
const pointSizeSlider = document.getElementById('point-size');

let session = null;
let currentLeftData = null; // Original RGB data for colored PC
let currentLeftImgData = null; // Resized for model
let currentRightImgData = null;
let renderer3d = null;

const TARGET_WIDTH = 640;
const TARGET_HEIGHT = 480;

// Helper: Update Status
const updateStatus = (msg) => {
  statusBar.textContent = `Status: ${msg}`;
  console.log(msg);
};

// Helper: Load Image
const loadImage = (url) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
};

const preprocess = (img) => {
  const canvas = document.createElement('canvas');
  canvas.width = TARGET_WIDTH;
  canvas.height = TARGET_HEIGHT;
  const ctx = canvas.getContext('2d');

  const srcW = img.width / 2;
  const srcH = img.height;

  // Process Left
  ctx.drawImage(img, 0, 0, srcW, srcH, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  const leftData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  imgLeft.src = canvas.toDataURL();

  // Process Right
  ctx.drawImage(img, srcW, 0, srcW, srcH, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  const rightData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  imgRight.src = canvas.toDataURL();

  return { leftData, rightData };
};

const toTensor = (imageData) => {
  const { data, width, height } = imageData;
  const floatData = new Float32Array(3 * width * height);

  for (let i = 0; i < width * height; i++) {
    floatData[i] = data[i * 4];
    floatData[width * height + i] = data[i * 4 + 1];
    floatData[2 * width * height + i] = data[i * 4 + 2];
  }

  return new ort.Tensor('float32', floatData, [1, 3, height, width]);
};

// Simple Turbo-like colormap
const getTurboColor = (v) => {
  // v in [0, 1]
  const r = Math.max(0, Math.min(255, Math.floor(255 * (v < 0.5 ? 4 * v * v : -4 * (v - 1) * (v - 1) + 1))));
  const g = Math.max(0, Math.min(255, Math.floor(255 * (v < 0.5 ? 2 * v : 2 * (1 - v)))));
  const b = Math.max(0, Math.min(255, Math.floor(255 * (v < 0.5 ? 1 - 2 * v : 0))));

  // Actually let's use a more accurate one or just HSL
  // HSL: 240 (blue) to 0 (red)
  const hue = (1.0 - v) * 240;
  return hue;
};

// Initialize Application
const init = async () => {
  try {
    updateStatus("Initializing...");

    // Init 3D Renderer
    renderer3d = new PointCloudRenderer(canvas3d);

    if (!navigator.gpu) {
      updateStatus("Error: WebGPU not supported");
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      updateStatus("Error: No WebGPU adapter");
      return;
    }

    updateStatus("Loading LiteAnyStereo...");
    const option = { executionProviders: ['webgpu'] };
    session = await ort.InferenceSession.create(MODEL_PATH, option);

    updateStatus("Ready");
    await loadSample();
    runBtn.disabled = false;
  } catch (e) {
    updateStatus("Init Error: " + e.message);
    console.error(e);
  }
};

const loadSample = async () => {
  const filename = sampleSelect.value;
  const path = `./assets/${filename}`;
  updateStatus(`Loading ${filename}...`);

  try {
    const img = await loadImage(path);
    const result = preprocess(img);
    currentLeftImgData = result.leftData;
    currentRightImgData = result.rightData;
    updateStatus("Sample loaded");

    // Clear disparity
    const ctx = canvasResult.getContext('2d');
    ctx.clearRect(0, 0, canvasResult.width, canvasResult.height);

  } catch (e) {
    updateStatus("Load Error");
    console.error(e);
  }
};

const runInference = async () => {
  if (!session || !currentLeftImgData || !currentRightImgData) return;

  updateStatus("Computing...");
  loadingOverlay.classList.remove('hidden');

  try {
    const leftTensor = toTensor(currentLeftImgData);
    const rightTensor = toTensor(currentRightImgData);

    const start = performance.now();
    const results = await session.run({ left: leftTensor, right: rightTensor });
    const end = performance.now();

    const dispTensor = results.disparity;
    const { data: dispData, dims } = dispTensor;
    const [B, C, H, W] = dims;

    updateStatus(`Done: ${(end - start).toFixed(1)}ms`);

    // Render 2D Disparity with colormap
    renderDisparity(dispData, H, W);

    // Update 3D Point Cloud
    renderer3d.update(currentLeftImgData.data, dispData, W, H);

  } catch (e) {
    updateStatus("Inference Error: " + e.message);
    console.error(e);
  } finally {
    loadingOverlay.classList.add('hidden');
  }
};

const renderDisparity = (data, height, width) => {
  canvasResult.width = width;
  canvasResult.height = height;
  const ctx = canvasResult.getContext('2d');
  const imgData = ctx.createImageData(width, height);

  let min = 1000, max = -1000;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  for (let i = 0; i < width * height; i++) {
    const val = data[i];
    let norm = (val - min) / (max - min);
    if (isNaN(norm)) norm = 0;

    // HSL to RGB conversion for better visualization
    const hue = (1.0 - norm) * 240;

    // Quick HSL to RGB for canvas
    // Or just use a simple Jet-like mapping
    let r, g, b;
    if (norm < 0.25) {
      r = 0; g = Math.floor(norm * 4 * 255); b = 255;
    } else if (norm < 0.5) {
      r = 0; g = 255; b = Math.floor((1 - (norm - 0.25) * 4) * 255);
    } else if (norm < 0.75) {
      r = Math.floor((norm - 0.5) * 4 * 255); g = 255; b = 0;
    } else {
      r = 255; g = Math.floor((1 - (norm - 0.75) * 4) * 255); b = 0;
    }

    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
};

sampleSelect.addEventListener('change', loadSample);
runBtn.addEventListener('click', runInference);
pointSizeSlider.addEventListener('input', (e) => {
  if (renderer3d) {
    renderer3d.setPointSize(parseFloat(e.target.value));
  }
});

init();
