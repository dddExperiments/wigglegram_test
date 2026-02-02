"""
Wigglegram Generator

Takes two stereo images and matched SIFT keypoints, performs stereo rectification,
and generates an animated GIF alternating between the rectified views.
"""

import base64
import io
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available")


def decode_image(base64_str):
    """Decode base64 image string to numpy array"""
    img_data = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def encode_gif(frames, duration_ms=200):
    """Encode list of numpy images to GIF base64 string"""
    pil_frames = []
    for frame in frames:
        # Convert BGR to RGB for PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0  # Infinite loop
    )
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def match_keypoints(keypoints_a, keypoints_b, ratio_threshold=0.75):
    """Match keypoints between two images using FLANN"""
    if not HAS_CV2:
        return [], []
    
    # Extract descriptors
    desc_a = np.array([kp['descriptor'] for kp in keypoints_a], dtype=np.float32)
    desc_b = np.array([kp['descriptor'] for kp in keypoints_b], dtype=np.float32)
    
    if len(desc_a) < 2 or len(desc_b) < 2:
        return [], []
    
    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc_a, desc_b, k=2)
    
    # Apply ratio test
    pts_a = []
    pts_b = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                pts_a.append([keypoints_a[m.queryIdx]['x'], keypoints_a[m.queryIdx]['y']])
                pts_b.append([keypoints_b[m.trainIdx]['x'], keypoints_b[m.trainIdx]['y']])
    
    return np.array(pts_a, dtype=np.float32), np.array(pts_b, dtype=np.float32)


def compute_homography(pts_a, pts_b, min_matches=4):
    """Compute homography from matched points using RANSAC"""
    if len(pts_a) < min_matches:
        return None
    
    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    return H


def rectify_images(img_a, img_b, pts_a, pts_b):
    """
    Rectify stereo images for wigglegram viewing.
    Uses homography to align image B to image A's coordinate frame.
    """
    h, w = img_a.shape[:2]
    
    # Compute homography (B -> A)
    H = compute_homography(pts_a, pts_b)
    
    if H is None:
        # Fallback: just resize to same dimensions
        print("Warning: Not enough matches for homography, using identity")
        img_b_rectified = cv2.resize(img_b, (w, h))
        return img_a, img_b_rectified, 0
    
    # Warp image B to align with image A
    img_b_rectified = cv2.warpPerspective(img_b, H, (w, h))
    
    # Count inliers
    pts_b_warped = cv2.perspectiveTransform(pts_b.reshape(-1, 1, 2), H).reshape(-1, 2)
    distances = np.sqrt(np.sum((pts_a - pts_b_warped) ** 2, axis=1))
    inliers = np.sum(distances < 5.0)
    
    return img_a, img_b_rectified, int(inliers)


def create_wigglegram(img_a_base64, img_b_base64, keypoints_a, keypoints_b, 
                      matches=None, max_size=800, frame_duration_ms=200):
    """
    Main function to create a wigglegram GIF.
    
    Args:
        img_a_base64: Base64-encoded image A
        img_b_base64: Base64-encoded image B  
        keypoints_a: List of keypoint dicts from image A
        keypoints_b: List of keypoint dicts from image B
        matches: Optional list of [idx_a, idx_b] lists. If provided, skips FLANN.
        max_size: Maximum dimension for output GIF
        frame_duration_ms: Duration of each frame in milliseconds
    
    Returns:
        dict with 'gif' (base64 GIF), 'matches' (count), or 'error'
    """
    if not HAS_CV2 or not HAS_PIL:
        return {"error": "OpenCV and PIL required for wigglegram generation"}
    
    try:
        # Decode images
        img_a = decode_image(img_a_base64)
        img_b = decode_image(img_b_base64)
        
        if img_a is None or img_b is None:
            return {"error": "Failed to decode images"}
        
        # Match keypoints
        if matches:
            # Reconstruct points from indices
            pts_a = []
            pts_b = []
            try:
                for idx_a, idx_b in matches:
                    pts_a.append([keypoints_a[idx_a]['x'], keypoints_a[idx_a]['y']])
                    pts_b.append([keypoints_b[idx_b]['x'], keypoints_b[idx_b]['y']])
                
                pts_a = np.array(pts_a, dtype=np.float32)
                pts_b = np.array(pts_b, dtype=np.float32)
            except (IndexError, TypeError, KeyError) as e:
                 return {"error": f"Invalid match indices: {e}"}
        else:
            # Use FLANN
            pts_a, pts_b = match_keypoints(keypoints_a, keypoints_b)
            
        match_count = len(pts_a)
        
        print(f"[Wigglegram] Matched {match_count} keypoints")
        
        if match_count < 4:
            return {"error": f"Only {match_count} matches found, need at least 4"}
        
        # Rectify
        rect_a, rect_b, inliers = rectify_images(img_a, img_b, pts_a, pts_b)
        
        print(f"[Wigglegram] Rectification complete, {inliers} inliers")
        
        # Resize for reasonable GIF size
        h, w = rect_a.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rect_a = cv2.resize(rect_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rect_b = cv2.resize(rect_b, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create GIF with both frames
        gif_base64 = encode_gif([rect_a, rect_b], duration_ms=frame_duration_ms)
        
        return {
            "gif": gif_base64,
            "matches": match_count,
            "inliers": inliers
        }
        
    except Exception as e:
        print(f"[Wigglegram] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
