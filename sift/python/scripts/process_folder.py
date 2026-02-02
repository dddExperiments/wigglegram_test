
import os
import sys
import argparse
import numpy as np
import time

# Add python module path (adjust if needed depending on where script is run)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import websiftgpu_py
except ImportError:
    # Try importing from Release/Debug folders if build locally
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Release')))
    try:
        import websiftgpu_py
    except ImportError:
        print("Error: Could not import websiftgpu_py. Please build the project first.")
        sys.exit(1)

try:
    from PIL import Image
    import cv2 
except ImportError:
    print("Error: This script requires PIL (Pillow) or OpenCV (opencv-python).")
    print("pip install Pillow opencv-python")
    # We can try to proceed if one is available but let's assume both or one.
    # Actually we will try to use PIL if cv2 missing, or vice versa.
    pass

def load_image(path, max_dim=0):
    """Load image as RGBA uint8 numpy array. Returns (img, restore_factor)."""
    restore_factor = 1.0
    
    if 'cv2' in sys.modules:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, 1.0
            
        # Get original dimensions (h, w)
        h, w = img.shape[:2]
        
        # Resize if needed
        if max_dim > 0:
            max_side = max(h, w)
            if max_side > max_dim:
                scale_down = max_dim / max_side
                new_w = int(w * scale_down)
                new_h = int(h * scale_down)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                restore_factor = 1.0 / scale_down
        
        # Convert to RGBA
        if len(img.shape) == 2: # Gray
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3: # RGB/BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[2] == 4: # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
        return img, restore_factor
        
    else:
        try:
            img_pil = Image.open(path)
            w, h = img_pil.size
            
            if max_dim > 0:
                max_side = max(h, w)
                if max_side > max_dim:
                    scale_down = max_dim / max_side
                    new_w = int(w * scale_down)
                    new_h = int(h * scale_down)
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    restore_factor = 1.0 / scale_down
                    print(f"  Resizing {w}x{h} -> {new_w}x{new_h} (scale: {scale_down:.4f})")

            img_pil = img_pil.convert('RGBA')
            return np.array(img_pil), restore_factor
        except Exception:
            return None, 1.0

def save_sift_format(filepath, keypoints):
    """Save keypoints in VisualSFM/Lowe ASCII format"""
    # Format: 
    # <count> 128
    # y x scale ori d0 ... d127 (Lowe's format matches x/y order of COLMAP/VisualSFM in practice)
    
    with open(filepath, 'w') as f:
        f.write(f"{len(keypoints)} 128\n")
        
        for kp in keypoints:
            x = kp['x']
            y = kp['y']
            scale = kp['scale']
            ori = kp['orientation']
            desc = kp.get('descriptor', np.zeros(128))
            
            # Normalize and clamp descriptor to [0, 255] integer range
            desc = np.array(desc, dtype=np.float32)
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                desc /= norm
            
            # Clip high peaks to reduce influence of large gradients
            np.clip(desc, 0, 0.2, out=desc)
            
            # Re-normalize
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                desc /= norm
            
            # Scale to byte [0, 255]
            desc = (desc * 512).clip(0, 255).astype(int)
            
            # Write keypoint line
            desc_str = " ".join(map(str, desc))
            f.write(f"{x:.4f} {y:.4f} {scale:.4f} {ori:.4f} {desc_str}\n")

def normalize_descriptors(descs_np):
    """Normalize descriptors using SIFT L2-clip-L2 logic."""
    # Normalize L2
    norms = np.linalg.norm(descs_np, axis=1, keepdims=True)
    # Avoid divide by zero
    norms[norms < 1e-6] = 1.0 
    descs_np /= norms
    
    # SIFT magic: clip to 0.2, re-normalize
    np.clip(descs_np, 0, 0.2, out=descs_np)
    norms = np.linalg.norm(descs_np, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    descs_np /= norms
    return descs_np

import concurrent.futures

def load_image_task(args):
    """Worker function for loading images in parallel."""
    path, max_dim = args
    return load_image(path, max_dim)

def load_sift_descriptors(filepath):
    """Load descriptors from a .sift file."""
    try:
        with open(filepath, 'r') as f:
            line = f.readline()
            if not line: return None
            
            parts = line.strip().split()
            count = int(parts[0])
            dim = int(parts[1])
            
            if count == 0:
                return np.zeros((0, 128), dtype=np.float32)
            
            descriptors = []
            for _ in range(count):
                line = f.readline()
                if not line: break
                vals = list(map(float, line.strip().split()))
                # Format: y x scale ori d0..d127
                # Descriptor starts at index 4
                desc = vals[4:]
                descriptors.append(desc)
                
            descs_np = np.array(descriptors, dtype=np.float32)
            return normalize_descriptors(descs_np)
    except Exception as e:
        print(f"Error loading descriptors from {filepath}: {e}")
        return None

def process_images(sift, files, input_dir, output_dir, max_dim=0, num_threads=4):
    """Detect features in images using parallel loading and sequential GPU processing."""
    total_time = 0
    start_total_time = time.time()
    count = 0
    valid_files = [] # Keep track of files successfully processed
    # Create a list of arguments for the worker
    load_args = [(os.path.join(input_dir, f), max_dim) for f in files]
    
    print(f"Processing {len(files)} images with {num_threads} processes for loading...")
    
    # Use ProcessPoolExecutor for true CPU parallelism (bypassing GIL)
    # This is crucial for operations like image resizing which might not fully release GIL or contend.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks. 
        future_to_file = {executor.submit(load_image_task, arg): f for arg, f in zip(load_args, files)}
        
        for future in concurrent.futures.as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                img, restore_factor = future.result()
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                continue
                
            if img is None:
                print(f"Failed to load {fname}")
                continue
                
            print(f"Processing {fname} ({img.shape[1]}x{img.shape[0]})...", end='', flush=True)
            
            # GPU Processing (Main Thread)
            t0 = time.time()
            kps = sift.detect(img)
            dt = time.time() - t0
            total_time += dt
            count += 1
            
            print(f" Found {len(kps)} features in {dt*1000:.1f}ms")
            
            # Restore keypoints to original scale if resized
            if restore_factor != 1.0:
                for kp in kps:
                    kp['x'] *= restore_factor
                    kp['y'] *= restore_factor
                    kp['scale'] *= restore_factor
            
            # Save
            base_name = os.path.splitext(fname)[0]
            sift_path = os.path.join(output_dir, base_name + ".sift")
            save_sift_format(sift_path, kps)
            valid_files.append(fname)

    if count > 0:
        print(f"Avg GPU Time: {total_time/count*1000:.1f}ms per image.")
        print(f"Total CPU Time: {time.time() - start_total_time:.4f}s")
        
    return valid_files

def perform_matching(files, output_dir, match_ratio=0.75, num_threads=4):
    """Perform exhaustive matching using GPU Matcher with parallel descriptor loading."""
    if len(files) < 2:
        return

    print(f"Performing exhaustive matching (Ratio: {match_ratio}) with {num_threads} threads for loading...")
    matches_path = os.path.join(output_dir, "matches.txt")
    
    # Init matcher
    try:
        matcher = websiftgpu_py.SIFTMatcher()
        matcher.init()
        print("Initialized GPU Matcher.")
    except Exception as e:
        print(f"Failed to init GPU Matcher: {e}")
        print("Falling back to CPU matching not implemented fully in this branch.")
        sys.exit(1)
    
    with open(matches_path, 'w') as f_out:
        # We need a persistent executor for the inner loop tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            
            for i in range(len(files)):
                name1 = files[i]
                base1 = os.path.splitext(name1)[0]
                path1 = os.path.join(output_dir, base1 + ".sift")
                
                # Load query descriptor (can be done on main or thread, main is fine for outer loop)
                desc1 = load_sift_descriptors(path1)
                
                if desc1 is None or len(desc1) == 0:
                    continue
                
                # Prepare tasks for all j > i
                future_to_index = {}
                for j in range(i + 1, len(files)):
                    name2 = files[j]
                    base2 = os.path.splitext(name2)[0]
                    path2 = os.path.join(output_dir, base2 + ".sift")
                    future = executor.submit(load_sift_descriptors, path2)
                    future_to_index[future] = j

                # Process results as they come in (Order doesn't strictly match j order, but pairs are unique)
                # If strict order in file is needed, we would wait differently, but for matches.txt usually any order is fine
                # as long as pairs are unique. However, standard convention usually follows loop order.
                # Let's enforce order to be deterministic.
                
                # To force order:
                # We can just iterate the list of futures we just created
                sorted_futures = list(future_to_index.keys()) 
                # Note: This means we wait for j=i+1, then j=i+2. Parallelism helps if loading is slow.
                
                for future in sorted_futures:
                    j = future_to_index[future]
                    name2 = files[j]
                    
                    try:
                        desc2 = future.result()
                    except Exception as e:
                        print(f"Error loading {name2}: {e}")
                        continue
                        
                    if desc2 is None or len(desc2) == 0:
                        continue

                    # GPU Match (Sequential)
                    matches = matcher.match(desc1, desc2, match_ratio)
                    
                    if len(matches) > 0:
                        valid_indices = matches[:, 0]
                        matched_targets = matches[:, 1]
                        
                        f_out.write(f"{name1}\n")
                        f_out.write(f"{name2}\n")
                        f_out.write(f"{len(matches)}\n")
                        
                        for k in range(len(matches)):
                            f_out.write(f"{valid_indices[k]} {matched_targets[k]}\n")
                        
                        f_out.write("\n") 
                        print(f"  Matches {name1} vs {name2}: {len(matches)}")


def main():
    if 'cv2' in sys.modules:
        cv2.setNumThreads(0) # Prevent OpenCV from spawning threads, avoiding contention with our Pool

    parser = argparse.ArgumentParser(description="Run WebGPU SIFT on a folder of images.")
    parser.add_argument("--input", required=True, help="Input folder container images")
    parser.add_argument("--output", help="Output folder (default: input folder)", default=None)
    
    parser.add_argument("--match_images", action="store_true", help="Perform exhaustive matching and save to matches.txt")
    parser.add_argument("--match_ratio", type=float, default=0.75, help="Distance ratio for matching (default: 0.75)")
    
    parser.add_argument("--max_dimension", type=int, default=0, help="Max image dimension for processing (0 = ignore). Resizes on CPU, scales Keypoints back.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for image loading and descriptor fetching (default: 4)")

    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output if args.output else input_dir
    
    if not os.path.isdir(input_dir):
        print(f"Input is not a directory: {input_dir}")
        sys.exit(1)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Initializing WebGPU SIFT...")
    try:
        sift = websiftgpu_py.SIFT()
    except Exception as e:
        print(f"Failed to init SIFT: {e}")
        sys.exit(1)
        
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process
    valid_files = process_images(sift, files, input_dir, output_dir, args.max_dimension, args.num_threads)
    
    # Match
    if args.match_images:
        perform_matching(valid_files, output_dir, args.match_ratio, args.num_threads)

if __name__ == "__main__":
    main()

