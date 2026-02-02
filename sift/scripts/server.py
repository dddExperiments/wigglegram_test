"""
Simple HTTP server for WebGPU SIFT application.

Serves static files and provides a /match endpoint for feature matching.
"""

import http.server
import socketserver
import json
import os
import sys
import threading

# Add parent directory to path to allow importing from demos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matcher import get_matcher
from demos.wigglegram import create_wigglegram

PORT = 8000

class SIFTHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS and matching endpoint"""
    
    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests including /stop endpoint"""
        if self.path == '/stop':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server shutting down...')
            print("\n[Server] Stop endpoint called, shutting down...")
            # Schedule shutdown in a separate thread to allow response to be sent
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            # Check if file exists (custom 404)
            path = self.translate_path(self.path)
            if not os.path.exists(path) and not self.path.startswith(('/match', '/detect', '/wigglegram')): 
                # Note: POST endpoints might come here if Method is wrong, but this is DO_GET. 
                # Pure static file check.
                self.send_error(404, f"File not found: {self.path}")
                print(f"[404] {self.path}")
                return
                
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/match':
            self.handle_match()
        elif self.path == '/match_pair':
            self.handle_match_pair()
        elif self.path == '/wigglegram':
            self.handle_wigglegram()
        elif self.path == '/detect':
            self.handle_detect()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_detect(self):
        """Handle SIFT detection request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            data = json.loads(post_data)
            image_data = data.get('image') # Base64 string
            
            if not image_data:
                self.send_json_response({"error": "No image provided"}, status=400)
                return

            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            import base64
            import numpy as np
            import cv2
            
            # Decode image
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.send_json_response({"error": "Failed to decode image"}, status=400)
                return

            # Run SIFT
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            # Format results
            results = []
            if keypoints is not None:
                for i, kp in enumerate(keypoints):
                    desc = descriptors[i].tolist() if descriptors is not None else []
                    results.append({
                        "x": float(kp.pt[0]),
                        "y": float(kp.pt[1]),
                        "scale": float(kp.size), # Map size to scale roughly
                        "orientation": float(np.deg2rad(kp.angle)), # Map deg to rad
                        "response": float(kp.response),
                        "octave": int(kp.octave),
                        "descriptor": desc
                    })
            
            print(f"[Detect] Processed image, found {len(results)} keypoints")
            self.send_json_response(results)

        except Exception as e:
            print(f"[Detect Error] {e}")
            import traceback
            traceback.print_exc()
            self.send_json_response({"error": str(e)}, status=500)

    def handle_match(self):
        """Handle single-image self-matching (for stereo pairs)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            keypoints = data.get('keypoints', [])
            print(f"[Match] Received {len(keypoints)} keypoints for self-matching")
            
            matcher = get_matcher()
            result = matcher.match_self(keypoints)
            
            print(f"[Match] Found {result.get('count', 0)} matches")
            
            self.send_json_response(result)
            
        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except Exception as e:
            print(f"[Match Error] {e}")
            self.send_json_response({"error": str(e)}, status=500)
    
    def handle_match_pair(self):
        """Handle two-image matching"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            keypoints_a = data.get('keypoints_a', [])
            keypoints_b = data.get('keypoints_b', [])
            print(f"[Match Pair] Received {len(keypoints_a)} + {len(keypoints_b)} keypoints")
            
            matcher = get_matcher()
            result = matcher.match(keypoints_a, keypoints_b)
            
            print(f"[Match Pair] Found {result.get('count', 0)} matches")
            
            self.send_json_response(result)
            
        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except Exception as e:
            print(f"[Match Pair Error] {e}")
            self.send_json_response({"error": str(e)}, status=500)
    
    def handle_wigglegram(self):
        """Handle wigglegram generation request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Use strict=False for control characters in base64 if needed
            data = json.loads(post_data)
            
            img_a = data.get('image_a')
            img_b = data.get('image_b')
            keypoints_a = data.get('keypoints_a', [])
            keypoints_a = data.get('keypoints_a', [])
            keypoints_b = data.get('keypoints_b', [])
            matches = data.get('matches', None) # Optional: client-provided matches
            
            print(f"[Wigglegram] Received request with {len(keypoints_a)} + {len(keypoints_b)} keypoints")
            if matches:
                 print(f"[Wigglegram] Using {len(matches)} client-provided matches")
            
            if not img_a or not img_b:
                self.send_json_response({"error": "Missing images"}, status=400)
                return
                
            result = create_wigglegram(img_a, img_b, keypoints_a, keypoints_b, matches=matches)
            
            if "error" in result:
                print(f"[Wigglegram Error] {result['error']}")
                self.send_json_response(result, status=400)
            else:
                print(f"[Wigglegram] Success! Matches: {result.get('matches')}, Inliers: {result.get('inliers')}")
                self.send_json_response(result)
                
        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except Exception as e:
            print(f"[Wigglegram Error] {e}")
            import traceback
            traceback.print_exc()
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data, status=200):
        """Send a JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom logging format"""
        if '/match' in args[0] if args else False:
            # Log match requests
            print(f"[Server] {args[0]}")
        elif '/detect' in args[0] if args else False:
             print(f"[Server] {args[0]}")
        elif not any(ext in args[0] for ext in ['.js', '.css', '.jpg', '.png', '.ico']):
            # Log non-static requests
            print(f"[Server] {args[0]}")


def main():
    print(f"""
╔═══════════════════════════════════════════╗
║         WebGPU SIFT Server                ║
╠═══════════════════════════════════════════╣
║  URL: http://localhost:{PORT}               ║
║                                           ║
║  Endpoints:                               ║
║    GET  /           - Web application     ║
║    GET  /stop       - Stop the server     ║
║    POST /match      - Self-matching       ║
║    POST /match_pair - Two-image matching  ║
║    POST /wigglegram - Generate GIF        ║
║                                           ║
║  Press Ctrl+C or visit /stop to stop     ║
╚═══════════════════════════════════════════╝
""")
    
    # Allow socket reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), SIFTHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")


if __name__ == "__main__":
    main()
