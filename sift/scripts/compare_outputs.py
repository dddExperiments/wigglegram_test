import json
import math
import sys
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def dist(k1, k2):
    return math.sqrt((k1['x'] - k2['x'])**2 + (k1['y'] - k2['y'])**2)

def compare(cpp_path, web_path):
    if not os.path.exists(cpp_path):
        print(f"Error: {cpp_path} not found.")
        return
    if not os.path.exists(web_path):
        print(f"Error: {web_path} not found.")
        return

    print(f"Loading {cpp_path} and {web_path}")
    cpp = load_json(cpp_path)
    web = load_json(web_path)
    
    kp_cpp = cpp['keypoints']
    desc_cpp = cpp['descriptors']
    
    kp_web = web['keypoints']
    desc_web = web['descriptors']
    
    print(f"C++ Keypoints: {len(kp_cpp)}")
    print(f"Web Keypoints: {len(kp_web)}")
    
    matches = 0
    total_desc_diff = 0.0
    matched_indices = set()
    
    for i, kc in enumerate(kp_cpp):
        best_dist = 1e9
        best_idx = -1
        
        for j, kw in enumerate(kp_web):
            if j in matched_indices: continue
            
            # Simple spatial check first
            if abs(kc['x'] - kw['x']) > 2.0 or abs(kc['y'] - kw['y']) > 2.0:
                continue
            
            # Compare Scale/Octave?
            # Keypoint matching should account for scale, but let's stick to location first
            
            d = dist(kc, kw)
            if d < best_dist:
                best_dist = d
                best_idx = j
                
        if best_dist < 1.0: # 1 pixel tolerance
            matches += 1
            matched_indices.add(best_idx)
            
            # Compare descriptors
            dc = desc_cpp[i]
            dw = desc_web[best_idx]
            
            # L2 distance
            dd = 0
            for v1, v2 in zip(dc, dw):
                dd += (v1 - v2)**2
            l2 = math.sqrt(dd)
            total_desc_diff += l2
            
    print("-" * 30)
    print(f"Matches found: {matches}")
    if len(kp_cpp) > 0:
        print(f"Match Rate (vs C++): {matches / len(kp_cpp) * 100:.2f}%")
        
    if matches > 0:
        avg_diff = total_desc_diff / matches
        print(f"Avg Descriptor L2 Distance: {avg_diff:.4f}")
        # Typical SIFT descriptor values are 0-255 or normalized.
        # If normalized to unit vector, max dist is 2.
        # If standard SIFT (0-255), dist can be larger.
        # Web implementation usually outputs normalized or 0-255 depending on shader.
        # We'll see.
        
    if matches < len(kp_cpp) * 0.8:
        print("WARNING: Match rate is low (< 80%)")
    else:
        print("SUCCESS: High match rate.")
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p1 = os.path.join(base, "verification", "cpp.json")
        p2 = os.path.join(base, "verification", "web.json")
    else:
        p1 = sys.argv[1]
        p2 = sys.argv[2]
        
    compare(p1, p2)
