import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export class PointCloudRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.init();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        const rect = this.canvas.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(75, rect.width / rect.height, 0.01, 100);
        this.camera.position.set(0, 0, 0.5);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.rotateSpeed = 0.5;

        // Helper: Grid (optional, but looks pro)
        // const grid = new THREE.GridHelper(2, 20, 0x333333, 0x111111);
        // grid.rotation.x = Math.PI / 2;
        // this.scene.add(grid);

        this.points = null;
        this.geometry = new THREE.BufferGeometry();
        this.material = new THREE.PointsMaterial({
            size: 0.003,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            sizeAttenuation: true
        });

        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        window.addEventListener('resize', () => this.onResize());
    }

    onResize() {
        const parent = this.canvas.parentElement;
        const width = parent.clientWidth;
        const height = parent.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    update(rgbData, dispData, width, height) {
        const numPoints = width * height;
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 3);

        // Params for re-projection (Generic values for visualization)
        const f = width * 1.0;
        const cx = width / 2;
        const cy = height / 2;
        const b = 0.1;

        let validCount = 0;

        for (let v = 0; v < height; v++) {
            for (let u = 0; u < width; u++) {
                const i = v * width + u;
                const d = dispData[i];

                if (d > 2.0) { // Threshold for noise
                    const z = (f * b) / d;
                    const x = ((u - cx) * z) / f;
                    const y = -((v - cy) * z) / f;

                    positions[validCount * 3] = x;
                    positions[validCount * 3 + 1] = y;
                    positions[validCount * 3 + 2] = -z;

                    colors[validCount * 3] = rgbData[i * 4] / 255;
                    colors[validCount * 3 + 1] = rgbData[i * 4 + 1] / 255;
                    colors[validCount * 3 + 2] = rgbData[i * 4 + 2] / 255;

                    validCount++;
                }
            }
        }

        const finalPositions = positions.slice(0, validCount * 3);
        const finalColors = colors.slice(0, validCount * 3);

        this.geometry.setAttribute('position', new THREE.BufferAttribute(finalPositions, 3));
        this.geometry.setAttribute('color', new THREE.BufferAttribute(finalColors, 3));

        // Explicitly flag for update
        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.color.needsUpdate = true;
        this.geometry.computeBoundingSphere(); // Important for frustum culling

        if (!this.points) {
            this.points = new THREE.Points(this.geometry, this.material);
            this.scene.add(this.points);
        }

        // Centering and Auto-camera (Reset view to look from between cameras)
        this.geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        this.geometry.boundingBox.getCenter(center);

        this.controls.target.copy(center);

        // Calculate appropriate camera distance based on bounding box size
        const size = new THREE.Vector3();
        this.geometry.boundingBox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        // Position camera back in Z and slightly up in Y
        this.camera.position.set(center.x, center.y + 0.1, center.z + (maxDim > 0 ? maxDim * 1.5 : 0.5));
        this.controls.update();
    }

    setPointSize(sliderValue) {
        if (this.material) {
            // Scale slider 1-100 to Three.js world size 0.0001-0.01
            this.material.size = sliderValue / 10000;
        }
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
