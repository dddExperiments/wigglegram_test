const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    console.log("Starting Vite...");
    // Use npx vite to avoid sticking to npm scripts if easier, but npm run dev is consistent
    const vite = spawn('npm.cmd', ['run', 'dev', '--', '--port', '5000'], {
        shell: true,
        stdio: 'pipe',
        cwd: path.resolve(__dirname, '..')
    });

    let viteOutput = '';
    vite.stdout.on('data', d => {
        const s = d.toString();
        viteOutput += s;
        // console.log(s); 
    });

    // Wait for "localhost" or specific port
    console.log("Waiting for Vite server...");
    const start = Date.now();
    while (!viteOutput.includes('http://localhost') && Date.now() - start < 30000) {
        await new Promise(r => setTimeout(r, 500));
    }

    if (!viteOutput.includes('http://localhost')) {
        console.error("Vite did not start in time. Output:\n" + viteOutput);
        process.exit(1);
    }

    console.log("Vite ready. Launching Browser...");
    // Get port from output if possible, assuming 5000 default
    const portMatch = viteOutput.match(/localhost:(\d+)/);
    const port = portMatch ? portMatch[1] : '5000';
    console.log(`Using port: ${port}`);

    let browser;
    try {
        console.log("Launching visible browser for WebGPU access...");
        browser = await chromium.launch({
            headless: false,
            args: ['--enable-unsafe-webgpu']
        });

        const page = await browser.newPage();

        // Log console messages
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        page.on('pageerror', err => console.error('PAGE ERROR:', err));

        const url = `http://localhost:${port}/tests/verification.html`;
        console.log(`Navigating to ${url}`);
        await page.goto(url);

        console.log("Waiting for #result...");
        await page.waitForFunction(() => {
            const el = document.getElementById('result');
            return el && el.textContent.includes('keypoints');
        }, null, { timeout: 60000 });

        const content = await page.textContent('#result');
        console.log(`Got content length: ${content.length}`);

        const outPath = path.resolve(__dirname, '../verification/web.json');
        fs.writeFileSync(outPath, content);
        console.log(`Saved to ${outPath}`);

    } catch (e) {
        console.error("Error during capture:", e);
        process.exit(1);
    } finally {
        if (browser) await browser.close();
        if (vite) {
            console.log("Killing vite...");
            spawn("taskkill", ["/pid", vite.pid, '/f', '/t']);
        }
        process.exit(0);
    }
}
main();
