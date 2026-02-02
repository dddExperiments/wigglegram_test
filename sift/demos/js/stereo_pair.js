import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { SIFTCPU } from '../../src/sift-cpu.js';
import { SIFTWebGPUDefault } from '../../src/sift-webgpu-default.js';
import { SIFTWebGPUPacked } from '../../src/sift-webgpu-packed.js';
import { WigglegramGenerator } from './wigglegram.js';

// Global state
let camera, scene, renderer;
let siftA, siftB;           // CPU SIFT instances
let siftWebGPUDefaultA, siftWebGPUDefaultB; // WebGPU Default instances
let siftWebGPUPackedA, siftWebGPUPackedB;   // WebGPU Packed instances
let imageQuadA, imageQuadB;
let textureA, textureB;
let useWebGPU = true;
let currentView = 'A';      // Which image is currently displayed

// To track successful SIFT runs
let lastRunResults = {
    A: { keypoints: [], width: 0, height: 0 },
    B: { keypoints: [], width: 0, height: 0 }
};

init();

async function init() {
    if (!navigator.gpu) {
        document.body.innerHTML = '<div style="color:red; margin:20px;">WebGPU not supported!</div>';
        return;
    }

    const container = document.createElement('div');
    container.id = 'container';
    document.body.appendChild(container);

    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    renderer = new WebGPURenderer({ antialias: true, forceWebGL: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setAnimationLoop(animate);
    container.appendChild(renderer.domElement);

    // Initialize SIFT instances
    siftA = new SIFTCPU();
    siftB = new SIFTCPU();

    siftWebGPUDefaultA = new SIFTWebGPUDefault();
    siftWebGPUDefaultB = new SIFTWebGPUDefault();

    siftWebGPUPackedA = new SIFTWebGPUPacked();
    siftWebGPUPackedB = new SIFTWebGPUPacked();

    // Initialize WebGPU
    try {
        await Promise.all([
            siftWebGPUDefaultA.init(),
            siftWebGPUDefaultB.init(),
            siftWebGPUPackedA.init(),
            siftWebGPUPackedB.init()
        ]);
        console.log("WebGPU SIFT initialized");
    } catch (err) {
        console.warn("WebGPU init failed:", err);
        useWebGPU = false;
        document.getElementById('modeToggle').checked = false;
    }

    // Load initial images
    await loadImages();

    // Event Listeners
    window.addEventListener('resize', onWindowResize);
    setupEventListeners();

    updateStatus(`Ready (${useWebGPU ? 'WebGPU' : 'CPU'} mode)`);
}

async function loadImages() {
    const urlA = document.getElementById('imageSelectA').value;
    const urlB = document.getElementById('imageSelectB').value;

    updateStatus('Loading images...');

    // Load base images
    const [imgA, imgB] = await Promise.all([
        loadImageElement(urlA),
        loadImageElement(urlB)
    ]);

    // Create Textures for display
    if (imgA) textureA = createTexture(imgA);
    if (imgB) textureB = createTexture(imgB);

    // Load into SIFT instances
    // Create Bitmaps for WebGPU
    const bmpA = await createImageBitmap(imgA);
    const bmpB = await createImageBitmap(imgB);

    const promises = [
        siftA.loadImage(imgA),
        siftB.loadImage(imgB),
        // WebGPU loads (load into ALL variants)
        siftWebGPUDefaultA.device ? siftWebGPUDefaultA.loadImage(bmpA) : Promise.resolve(),
        siftWebGPUDefaultB.device ? siftWebGPUDefaultB.loadImage(bmpB) : Promise.resolve(),
        siftWebGPUPackedA.device ? siftWebGPUPackedA.loadImage(bmpA) : Promise.resolve(),
        siftWebGPUPackedB.device ? siftWebGPUPackedB.loadImage(bmpB) : Promise.resolve()
    ];

    await Promise.all(promises);

    // Bitmaps are cloned inside loadImage usually? Or processed immediately.
    // The implementations (check code) might keep reference or texture.
    // CPU SIFT processes immediately.
    // GPU SIFT processes to texture immediately.
    // So we can close bitmaps? Better safe to keep until run? No, typically upload and done.
    // But checking implementation:
    // SIFTWebGPU: queue.copyExternalImageToTexture. Safe to close.
    bmpA.close();
    bmpB.close();

    displayImage(currentView === 'A' ? textureA : textureB);
    clearKeypoints();

    // Reset last run results
    lastRunResults.A = { keypoints: [], width: imgA.width, height: imgA.height };
    lastRunResults.B = { keypoints: [], width: imgB.width, height: imgB.height };
    updateKeypointCounts();

    updateStatus(`Ready (${useWebGPU ? 'WebGPU' : 'CPU'} mode)`);
}

function createTexture(img) {
    const texture = new THREE.Texture(img);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    return texture;
}

async function loadImageElement(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

function setupEventListeners() {
    // Image selection
    document.getElementById('imageSelectA').addEventListener('change', loadImages);
    document.getElementById('imageSelectB').addEventListener('change', loadImages);

    // Toggles
    const gpuToggle = document.getElementById('modeToggle');
    const gpuMatchToggle = document.getElementById('gpuMatchToggle');
    const gpuOptToggle = document.getElementById('gpuOptToggle');

    // Initial state
    useWebGPU = gpuToggle.checked;

    function updateSiftSettings() {
        useWebGPU = gpuToggle.checked;
        const useOpt = gpuOptToggle.checked;
        updateStatus(`Ready (${useWebGPU ? 'WebGPU' : 'CPU'} mode, Optimized=${useOpt})`);
    }

    gpuToggle.addEventListener('change', updateSiftSettings);
    gpuOptToggle.addEventListener('change', updateSiftSettings);

    // Init status
    updateSiftSettings();

    // View toggle
    document.getElementById('viewToggle').addEventListener('change', (e) => {
        currentView = e.target.checked ? 'B' : 'A';
        displayImage(currentView === 'A' ? textureA : textureB);
        visualizeCurrentKeypoints();
    });

    // Run SIFT on both
    document.getElementById('runSiftBothBtn').addEventListener('click', runSiftOnBoth);

    // Generate wigglegram
    document.getElementById('generateWiggleBtn').addEventListener('click', generateWigglegram);

    // Preview
    document.getElementById('closePreviewBtn').addEventListener('click', closePreview);
    document.getElementById('overlay').addEventListener('click', closePreview);
}

function getActiveSiftInstances() {
    const useOpt = document.getElementById('gpuOptToggle').checked;

    if (useWebGPU) {
        if (useOpt) {
            return { A: siftWebGPUPackedA, B: siftWebGPUPackedB, type: 'GPU (Packed)' };
        } else {
            return { A: siftWebGPUDefaultA, B: siftWebGPUDefaultB, type: 'GPU (Default)' };
        }
    } else {
        return { A: siftA, B: siftB, type: 'CPU' };
    }
}

async function runSiftOnBoth() {
    const btn = document.getElementById('runSiftBothBtn');
    const wiggleBtn = document.getElementById('generateWiggleBtn');
    btn.disabled = true;
    wiggleBtn.disabled = true;

    try {
        const { A: activeSiftA, B: activeSiftB, type: mode } = getActiveSiftInstances();

        updateStatus(`Running SIFT on Image A (${mode})...`);
        const startA = performance.now();
        const kpsA = await activeSiftA.detectAndCompute(textureA.image);
        const timeA = performance.now() - startA;

        updateStatus(`Running SIFT on Image B (${mode})...`);
        const startB = performance.now();
        const kpsB = await activeSiftB.detectAndCompute(textureB.image);
        const timeB = performance.now() - startB;

        // Store results for visualization/wigglegram
        lastRunResults.A.keypoints = kpsA;
        lastRunResults.B.keypoints = kpsB;

        updateKeypointCounts();
        visualizeCurrentKeypoints();

        updateStatus(`SIFT complete: A=${kpsA.length} (${timeA.toFixed(0)}ms), B=${kpsB.length} (${timeB.toFixed(0)}ms)`);

        if (kpsA.length > 0 && kpsB.length > 0) {
            wiggleBtn.disabled = false;
        }

    } catch (err) {
        console.error(err);
        updateStatus(`Error: ${err.message}`);
    }

    btn.disabled = false;
}

function updateKeypointCounts() {
    document.getElementById('countA').textContent = lastRunResults.A.keypoints.length || '-';
    document.getElementById('countB').textContent = lastRunResults.B.keypoints.length || '-';
}

function visualizeCurrentKeypoints() {
    const data = currentView === 'A' ? lastRunResults.A : lastRunResults.B;
    if (data.keypoints && data.keypoints.length > 0) {
        visualizeKeypoints(data.keypoints, data.width, data.height);
    } else {
        clearKeypoints();
    }
}

function visualizeKeypoints(keypoints, imgWidth, imgHeight) {
    let canvas = document.getElementById('keypointCanvas');
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate display transform same as displayImage
    const imgAspect = imgWidth / imgHeight;
    const winAspect = canvas.width / canvas.height;
    let scale, offsetX, offsetY;

    if (imgAspect > winAspect) {
        scale = canvas.width / imgWidth;
        offsetX = 0;
        offsetY = (canvas.height - imgHeight * scale) / 2;
    } else {
        scale = canvas.height / imgHeight;
        offsetX = (canvas.width - imgWidth * scale) / 2;
        offsetY = 0;
    }

    // Draw keypoints
    // Optimizing drawing for large counts
    ctx.lineWidth = 1.0;

    keypoints.forEach(kp => {
        const x = offsetX + kp.x * scale;
        const y = offsetY + kp.y * scale;
        const r = Math.max(2, kp.sigma * scale * 1.0); // Adjust circle size

        // Color by octave (approx)
        const hue = (kp.octave / 4) * 120 + 100; // 4 octaves
        ctx.strokeStyle = `hsl(${hue}, 100%, 50%)`;

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();

        // Orientation
        if (kp.orientation !== undefined) {
            const endX = x + Math.cos(kp.orientation) * r;
            const endY = y + Math.sin(kp.orientation) * r;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(endX, endY);
            ctx.stroke();
        }
    });
}

function clearKeypoints() {
    const canvas = document.getElementById('keypointCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

async function generateWigglegram() {
    const btn = document.getElementById('generateWiggleBtn');
    btn.disabled = true;
    updateStatus('Generating wigglegram...');

    try {
        const kpsA = lastRunResults.A.keypoints;
        const kpsB = lastRunResults.B.keypoints;

        // Validation
        const validA = kpsA.filter(kp => kp.descriptor && kp.descriptor.length === 128);
        const validB = kpsB.filter(kp => kp.descriptor && kp.descriptor.length === 128);

        if (validA.length < 4 || validB.length < 4) {
            throw new Error(`Not enough keypoints with descriptors: A=${validA.length}, B=${validB.length}`);
        }

        // Matching
        let matches = null;
        const useGPUMatch = document.getElementById('gpuMatchToggle').checked;
        const { A: activeSiftA } = getActiveSiftInstances(); // For GPU matcher capability

        if (useGPUMatch && useWebGPU && activeSiftA.matchDescriptors) {
            updateStatus('Matching descriptors on GPU...');

            const flatten = (kps) => {
                const arr = new Float32Array(kps.length * 128);
                kps.forEach((kp, i) => arr.set(kp.descriptor, i * 128));
                return arr;
            };

            const flatA = flatten(validA);
            const flatB = flatten(validB);

            // Use Packed matcher (available on packed instance usually, but unpacked might support it too if we exposed it)
            // Let's assume activeSiftA has matchDescriptors. SIFTCPU does NOT.
            // SIFTWebGPUPacked has matchDescriptors. SIFTWebGPUDefault should too?
            // Actually, matchDescriptors is in 'js/matcher-webgpu.js'. SIFT classes might not expose it directly?
            // SIFTWebGPUPacked (from Step 68 import) might not.
            // Wait, previous `main.js` called `siftWebGPUA.matchDescriptors`.
            // Let's check `sift-webgpu-packed.js` imports.

            // If activeSiftA doesn't have it, we might need a standalone matcher.
            // But lets try calling it if exists.

            if (typeof activeSiftA.matchDescriptors === 'function') {
                const t0 = performance.now();
                matches = await activeSiftA.matchDescriptors(flatA, flatB);
                const timeMatch = performance.now() - t0;
                console.log(`GPU Match: ${matches.length} matches in ${timeMatch.toFixed(2)}ms`);
            } else {
                console.warn("GPU Matcher not available on current instance, falling back.");
            }
        }

        if (!matches || matches.length < 4) {
            updateStatus('Matching descriptors (brute-force)...');
            matches = bruteForceMatch(validA, validB, 0.75);
            console.log(`Brute-force Match: ${matches.length} matches`);
        }

        if (matches.length < 4) {
            throw new Error(`Only ${matches.length} matches found, need at least 4`);
        }

        // Browser-based wigglegram generation
        updateStatus('Computing homography (browser RANSAC)...');

        const urlA = document.getElementById('imageSelectA').value;
        const urlB = document.getElementById('imageSelectB').value;

        const [imgA, imgB] = await Promise.all([
            loadImageBitmap(urlA),
            loadImageBitmap(urlB)
        ]);

        const t0 = performance.now();
        const result = await WigglegramGenerator.create(imgA, imgB, validA, validB, matches, {
            maxSize: 600,
            frameDelay: 20
        });
        const totalTime = performance.now() - t0;

        if (result.error) {
            throw new Error(result.error);
        }

        showWigglegramBlob(result.gifBlob);
        updateStatus(`Wigglegram generated! ${result.matches} matches, ${result.inliers} inliers (${totalTime.toFixed(0)}ms)`);

    } catch (err) {
        console.error(err);
        updateStatus(`Error: ${err.message}`);
    }

    btn.disabled = false;
}

function bruteForceMatch(kpsA, kpsB, ratioThreshold = 0.75) {
    const matches = [];
    for (let i = 0; i < kpsA.length; i++) {
        const descA = kpsA[i].descriptor;
        let best = Infinity, secondBest = Infinity;
        let bestIdx = -1;

        for (let j = 0; j < kpsB.length; j++) {
            const descB = kpsB[j].descriptor;
            let dist = 0;
            for (let k = 0; k < 128; k++) {
                const d = descA[k] - descB[k];
                dist += d * d;
            }
            if (dist < best) {
                secondBest = best;
                best = dist;
                bestIdx = j;
            } else if (dist < secondBest) {
                secondBest = dist;
            }
        }

        // Ratio test (squared distances: best < ratioSq * secondBest ? No, usual ratio test is dist < ratio * secondDist)
        // Here dist is Squared? No, d*d. So dist is squared distance.
        // Ratio test on squared distance: best < ratio*ratio * secondBest.
        if (best < ratioThreshold * ratioThreshold * secondBest) {
            matches.push([i, bestIdx]);
        }
    }
    return matches;
}

async function loadImageBitmap(url) {
    const response = await fetch(url);
    const blob = await response.blob();
    return createImageBitmap(blob);
}

function showWigglegramBlob(blob) {
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('wigglegram-img');
    const downloadBtn = document.getElementById('downloadBtn');
    const overlay = document.getElementById('wigglegram-preview').parentElement.querySelector('#wigglegram-preview'); // Preview div
    const overlayBg = document.getElementById('overlay');

    img.src = url;
    downloadBtn.href = url;
    downloadBtn.download = 'wigglegram.gif';
    overlay.style.display = 'block';
    overlayBg.style.display = 'block';
}

function closePreview() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('wigglegram-preview').style.display = 'none';
}

function displayImage(texture) {
    if (imageQuadA) {
        scene.remove(imageQuadA);
        imageQuadA.geometry.dispose();
        imageQuadA.material.dispose();
    }

    const imgAspect = texture.image.width / texture.image.height;
    const winAspect = window.innerWidth / window.innerHeight;

    let scaleX, scaleY;
    if (imgAspect > winAspect) {
        scaleX = 1;
        scaleY = winAspect / imgAspect;
    } else {
        scaleX = imgAspect / winAspect;
        scaleY = 1;
    }

    const geometry = new THREE.PlaneGeometry(2 * scaleX, 2 * scaleY);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    imageQuadA = new THREE.Mesh(geometry, material);
    scene.add(imageQuadA);
}

function updateStatus(msg) {
    document.getElementById('status').textContent = msg;
    console.log('[Status]', msg);
}

function onWindowResize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    displayImage(currentView === 'A' ? textureA : textureB);
    visualizeCurrentKeypoints();
}

function animate() {
    renderer.render(scene, camera);
}
