"""
SIFT Feature Matcher using OpenCV FLANN

Accepts keypoints and descriptors from the web client
and performs matching using OpenCV's FLANN-based matcher.
"""

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Matching will return empty results.")


class SIFTMatcher:
    """FLANN-based matcher for SIFT descriptors"""
    
    def __init__(self, ratio_threshold=0.75):
        self.ratio_threshold = ratio_threshold
        
        if HAS_OPENCV:
            # FLANN parameters for SIFT (float descriptors)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.flann = None
    
    def match(self, keypoints_a, keypoints_b):
        """
        Match keypoints between two images.
        
        Args:
            keypoints_a: List of keypoint dicts from image A
            keypoints_b: List of keypoint dicts from image B
            
        Returns:
            List of match dicts with indices and distance
        """
        if not HAS_OPENCV:
            return {"matches": [], "error": "OpenCV not installed"}
        
        if not keypoints_a or not keypoints_b:
            return {"matches": [], "error": "Empty keypoint list"}
        
        # Extract descriptors as numpy arrays
        try:
            desc_a = np.array([kp['descriptor'] for kp in keypoints_a], dtype=np.float32)
            desc_b = np.array([kp['descriptor'] for kp in keypoints_b], dtype=np.float32)
        except (KeyError, TypeError) as e:
            return {"matches": [], "error": f"Invalid descriptor format: {e}"}
        
        if len(desc_a) < 2 or len(desc_b) < 2:
            return {"matches": [], "error": "Need at least 2 keypoints per image"}
        
        # Perform KNN matching (k=2 for ratio test)
        try:
            matches = self.flann.knnMatch(desc_a, desc_b, k=2)
        except cv2.error as e:
            return {"matches": [], "error": f"FLANN error: {e}"}
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append({
                        "idx_a": m.queryIdx,
                        "idx_b": m.trainIdx,
                        "distance": float(m.distance),
                        "pt_a": {
                            "x": keypoints_a[m.queryIdx]['x'],
                            "y": keypoints_a[m.queryIdx]['y']
                        },
                        "pt_b": {
                            "x": keypoints_b[m.trainIdx]['x'],
                            "y": keypoints_b[m.trainIdx]['y']
                        }
                    })
        
        return {
            "matches": good_matches,
            "count": len(good_matches),
            "total_a": len(keypoints_a),
            "total_b": len(keypoints_b)
        }
    
    def match_self(self, keypoints):
        """
        Match keypoints to themselves (for stereo pair detection).
        Useful when left and right views are in the same image.
        
        Returns matches excluding self-matches.
        """
        if not HAS_OPENCV:
            return {"matches": [], "error": "OpenCV not installed"}
        
        if not keypoints or len(keypoints) < 4:
            return {"matches": [], "error": "Need at least 4 keypoints"}
        
        desc = np.array([kp['descriptor'] for kp in keypoints], dtype=np.float32)
        
        # KNN with k=3 to skip self-match
        matches = self.flann.knnMatch(desc, desc, k=3)
        
        good_matches = []
        for match_triple in matches:
            if len(match_triple) >= 3:
                # Skip first match (self), apply ratio test on 2nd and 3rd
                m, n = match_triple[1], match_triple[2]
                if m.distance < self.ratio_threshold * n.distance:
                    # Additional filter: matches should be horizontally aligned (stereo)
                    pt_a = keypoints[m.queryIdx]
                    pt_b = keypoints[m.trainIdx]
                    
                    # Check vertical alignment (within 10% of image height tolerance)
                    y_diff = abs(pt_a['y'] - pt_b['y'])
                    if y_diff < 50:  # Adjust threshold as needed
                        good_matches.append({
                            "idx_a": m.queryIdx,
                            "idx_b": m.trainIdx,
                            "distance": float(m.distance),
                            "pt_a": {"x": pt_a['x'], "y": pt_a['y']},
                            "pt_b": {"x": pt_b['x'], "y": pt_b['y']}
                        })
        
        return {
            "matches": good_matches,
            "count": len(good_matches),
            "total": len(keypoints)
        }


# Singleton instance
_matcher = None

def get_matcher():
    global _matcher
    if _matcher is None:
        _matcher = SIFTMatcher()
    return _matcher
