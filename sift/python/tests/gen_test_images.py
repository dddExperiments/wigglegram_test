
import os
import numpy as np
try:
    from PIL import Image
except ImportError:
    print("PIL not found, cannot generate images.")
    exit(1)

out_dir = os.path.join(os.path.dirname(__file__), "test_images")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Create a simple pattern
w, h = 512, 512
img = np.zeros((h, w, 3), dtype=np.uint8)
img[:, :] = [50, 50, 50]
# Draw some squares
for i in range(10):
    x = np.random.randint(0, w-50)
    y = np.random.randint(0, h-50)
    color = np.random.randint(0, 255, 3)
    img[y:y+50, x:x+50] = color

img_pil = Image.fromarray(img)
img_pil.save(os.path.join(out_dir, "test1.jpg"))

print("Generated test1.jpg")
