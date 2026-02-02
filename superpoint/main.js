
// Helper to load image
async function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// Preprocess image for SuperPoint
function preprocess(img, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const { data } = imageData;
    const float32Data = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        float32Data[i] = gray;
    }

    return {
        tensor: new ort.Tensor('float32', float32Data, [1, 1, height, width]),
    };
}

const statusElement = document.getElementById('status');
const matchBtn = document.getElementById('matchBtn');
const matchCanvas = document.getElementById('matchCanvas');
const matchCtx = matchCanvas.getContext('2d');

let spSession;
let lgSession;

// Store extracted data for each slot
const slotData = [null, null];

async function init() {
    try {
        statusElement.textContent = 'Loading models...';

        // Load SuperPoint
        spSession = await ort.InferenceSession.create('./weights/superpoint.onnx', {
            executionProviders: ['webgpu'],
        });

        // Load LightGlue
        lgSession = await ort.InferenceSession.create('./weights/superpoint_lightglue.onnx', {
            executionProviders: ['webgpu'],
        });

        statusElement.textContent = 'Models loaded. Select images.';
    } catch (e) {
        statusElement.textContent = `Failed to load models: ${e.message}`;
        console.error(e);
    }
}

async function processSlot(file, slotIndex) {
    console.log(`Processing slot ${slotIndex}...`);
    statusElement.textContent = `Processing image ${Number(slotIndex) + 1}...`;

    try {
        const imageUrl = URL.createObjectURL(file);
        const img = await loadImage(imageUrl);

        let w = img.width;
        let h = img.height;
        w = Math.round(w / 8) * 8;
        h = Math.round(h / 8) * 8;

        const canvas = document.querySelector(`.outputCanvas[data-slot="${slotIndex}"]`);
        const ctx = canvas.getContext('2d');
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0, w, h);

        const { tensor } = preprocess(img, w, h);

        const t0 = performance.now();
        const results = await spSession.run({ 'image': tensor });
        const t1 = performance.now();

        const kpTensor = results['keypoints'];
        const scoresTensor = results['scores'];
        const descTensor = results['descriptors'];

        if (!kpTensor || !scoresTensor || !descTensor) {
            throw new Error(`SuperPoint outputs missing. Found: ${Object.keys(results)}`);
        }

        // Store data for matching
        slotData[slotIndex] = {
            image: img,
            width: w,
            height: h,
            keypoints: kpTensor,
            scores: scoresTensor,
            descriptors: descTensor
        };

        // Visualize keypoints
        const kps = kpTensor.data;
        const scores = scoresTensor.data;
        const numKpts = kpTensor.dims[1];

        ctx.fillStyle = '#00ff00';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;

        let validPoints = 0;
        for (let i = 0; i < numKpts; i++) {
            const x = Number(kps[i * 2]);
            const y = Number(kps[i * 2 + 1]);
            const score = scores[i];
            if (score <= 0.01) continue;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            validPoints++;
        }

        console.log(`[Slot ${slotIndex}] Extracted ${validPoints} keypoints in ${(t1 - t0).toFixed(2)}ms`);
        statusElement.textContent = `Image ${Number(slotIndex) + 1} done. Found ${validPoints} keypoints.`;

        // Enable match button if both slots are filled
        if (slotData[0] && slotData[1]) {
            matchBtn.disabled = false;
        }
    } catch (e) {
        statusElement.textContent = `Error in slot ${slotIndex}: ${e.message}`;
        console.error(e);
    }
}

async function runMatch() {
    if (!slotData[0] || !slotData[1]) return;

    statusElement.textContent = 'Running LightGlue matching...';
    matchBtn.disabled = true;

    try {
        // LightGlue expects normalized keypoints in range [-1, 1]
        // Formula: (x - w/2) / (max(w, h) / 2)
        const normalize = (kpts, w, h) => {
            const size = Math.max(w, h);
            const normKpts = new Float32Array(kpts.length);
            for (let i = 0; i < kpts.length / 2; i++) {
                normKpts[i * 2] = (Number(kpts[i * 2]) - w / 2) / (size / 2);
                normKpts[i * 2 + 1] = (Number(kpts[i * 2 + 1]) - h / 2) / (size / 2);
            }
            return normKpts;
        };

        const kpts0 = normalize(slotData[0].keypoints.data, slotData[0].width, slotData[0].height);
        const kpts1 = normalize(slotData[1].keypoints.data, slotData[1].width, slotData[1].height);

        const feeds = {
            'kpts0': new ort.Tensor('float32', kpts0, slotData[0].keypoints.dims),
            'kpts1': new ort.Tensor('float32', kpts1, slotData[1].keypoints.dims),
            'desc0': slotData[0].descriptors,
            'desc1': slotData[1].descriptors
        };

        const t0 = performance.now();
        const results = await lgSession.run(feeds);
        const t1 = performance.now();

        console.log('LightGlue Results:', Object.keys(results));

        const matches0Tensor = results['matches0'];
        const mscores0Tensor = results['mscores0'];

        if (!matches0Tensor) {
            throw new Error("LightGlue 'matches0' output missing.");
        }

        const matches0 = matches0Tensor.data;
        const numKpts0 = matches0Tensor.dims[1];
        const mscores0 = mscores0Tensor ? mscores0Tensor.data : null;

        // Count valid matches (often -1 for no match)
        let validMatchesCount = 0;
        const validMatchPairs = [];

        for (let i = 0; i < numKpts0; i++) {
            const matchIdx = Number(matches0[i]);
            if (matchIdx > -1) {
                validMatchesCount++;
                validMatchPairs.push({
                    idx0: i,
                    idx1: matchIdx,
                    score: mscores0 ? mscores0[i] : 1.0
                });
            }
        }

        console.log(`Matching took ${(t1 - t0).toFixed(2)}ms. Found ${validMatchesCount} matches.`);
        statusElement.textContent = `Found ${validMatchesCount} matches in ${(t1 - t0).toFixed(2)}ms.`;

        visualizeMatches(validMatchPairs);
    } catch (e) {
        statusElement.textContent = `Match Error: ${e.message}`;
        console.error(e);
    } finally {
        matchBtn.disabled = false;
    }
}

function visualizeMatches(matchPairs) {
    const s0 = slotData[0];
    const s1 = slotData[1];

    // Layout: Image 1 and Image 2 side-by-side
    const gap = 20;
    const totalWidth = s0.width + s1.width + gap;
    const maxHeight = Math.max(s0.height, s1.height);

    matchCanvas.width = totalWidth;
    matchCanvas.height = maxHeight;

    // Draw images
    matchCtx.fillStyle = '#111';
    matchCtx.fillRect(0, 0, totalWidth, maxHeight);

    matchCtx.drawImage(s0.image, 0, 0, s0.width, s0.height);
    matchCtx.drawImage(s1.image, s0.width + gap, 0, s1.width, s1.height);

    const kpts0 = s0.keypoints.data;
    const kpts1 = s1.keypoints.data;

    // Draw lines
    matchCtx.lineWidth = 1.5;

    matchPairs.forEach((m, i) => {
        const x0 = Number(kpts0[m.idx0 * 2]);
        const y0 = Number(kpts0[m.idx0 * 2 + 1]);
        const x1 = Number(kpts1[m.idx1 * 2]) + s0.width + gap;
        const y1 = Number(kpts1[m.idx1 * 2 + 1]);

        // Color based on index for variety
        const hue = (i * 137.5) % 360;
        matchCtx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.8)`;

        matchCtx.beginPath();
        matchCtx.moveTo(x0, y0);
        matchCtx.lineTo(x1, y1);
        matchCtx.stroke();

        // Draw endpoints
        matchCtx.fillStyle = '#fff';
        matchCtx.beginPath();
        matchCtx.arc(x0, y0, 2, 0, 2 * Math.PI);
        matchCtx.arc(x1, y1, 2, 0, 2 * Math.PI);
        matchCtx.fill();
    });
}

document.querySelectorAll('.imageInput').forEach(input => {
    input.addEventListener('change', async (e) => {
        if (!spSession) return;
        const file = e.target.files[0];
        if (!file) return;
        const slotIndex = e.target.dataset.slot;
        await processSlot(file, slotIndex);
    });
});

matchBtn.addEventListener('click', runMatch);

init();
