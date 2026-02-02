import numpy as np
import cv2
import os

os.makedirs('tests/temp_test', exist_ok=True)

# Create 800x600 image with strong features
img = np.zeros((600, 800, 3), dtype=np.uint8)
# Add random noise for texture
img[:] = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8) 
# Add patterns
for i in range(0, 800, 50):
    cv2.line(img, (i, 0), (i, 600), (255, 255, 255), 2)
for i in range(0, 600, 50):
    cv2.line(img, (0, i), (800, i), (255, 255, 255), 2)
    
cv2.circle(img, (400, 300), 100, (255, 0, 0), -1)

# Save duplicates
cv2.imwrite('tests/temp_test/img_A.png', img)
cv2.imwrite('tests/temp_test/img_B.png', img)
print("Created tests/temp_test/img_A.png and img_B.png")
