
print("Starting script...")
import cv2
print("cv2 imported")
import numpy as np
import os

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # Calculate new bounding box to avoid clipping
    abs_cos = abs(rot_mat[0,0])
    abs_sin = abs(rot_mat[0,1])
    bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
    rot_mat[0, 2] += bound_w/2 - image_center[0]
    rot_mat[1, 2] += bound_h/2 - image_center[1]
    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result

def run_test():
    print("Inside run_test")
    img_path = 'demo/book2.jpg'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    print(f"Reading {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image!")
        return
    
    print("Creating SIFT")
    # Initialize SIFT
    sift = cv2.SIFT_create()

    print("Detecting ref features")
    # Reference features
    kp1, des1 = sift.detectAndCompute(img, None)
    print(f"Reference keypoints: {len(kp1)}")

    angles = list(range(-180, 185, 5))
    inliers_counts = []

    # Matcher
    bf = cv2.BFMatcher()
    
    print("Angle,Inliers")

    for angle in angles:
        # Rotate
        img_rot = rotate_image(img, angle)
        
        # Detect
        kp2, des2 = sift.detectAndCompute(img_rot, None)
        
        val = 0
        if des2 is not None and len(kp2) >= 4:
            # Match
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 4:
                # RANSAC
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    val = np.sum(mask)
        
        inliers_counts.append(val)
        print(f"{angle},{val}")

    # Stats
    print(f"\nStats:")
    if inliers_counts:
        print(f"Max: {max(inliers_counts)}")
        print(f"Min: {min(inliers_counts)}")
        print(f"Avg: {sum(inliers_counts)/len(inliers_counts)}")

if __name__ == "__main__":
    print("Calling run_test")
    run_test()
