
import sys
import os

# Ensure the module can be found if in the current directory or nearby
# CMake output is likely in the same folder as this script due to our configuration
# OR we need to add the build/python folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import websiftgpu_py
except ImportError as e:
    print(f"Failed to import websiftgpu_py: {e}")
    sys.exit(1)

import numpy as np

def create_test_pattern(w, h):
    # Create RGBA pattern with a blob
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Fill slightly gray background
    img[:, :] = [20, 20, 20, 255]
    
    # Draw a white square in middle
    cw, ch = w//2, h//2
    size = 40
    img[ch-size:ch+size, cw-size:cw+size] = [255, 255, 255, 255]
    
    return img

def test_sift():
    print("Initializing SIFT...")
    sift = websiftgpu_py.SIFT()
    print("SIFT initialized.")
    
    w, h = 512, 512
    img = create_test_pattern(w, h)
    
    print(f"Running detection on {w}x{h} image...")
    kps = sift.detect(img)
    
    print(f"Found {len(kps)} keypoints.")
    
    if len(kps) > 0:
        print("First keypoint:", kps[0])
        print("PASS")
    else:
        print("No keypoints found (unexpected for this pattern).")
        print("FAIL")

if __name__ == "__main__":
    test_sift()
