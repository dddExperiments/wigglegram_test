import numpy as np
import cv2
import os

os.makedirs('tests/temp_test', exist_ok=True)
# Create 2000x1500 image
img = np.zeros((1500, 2000, 3), dtype=np.uint8)
# Add some patterns to detect features
for i in range(0, 2000, 100):
    cv2.line(img, (i, 0), (i, 1500), (255, 255, 255), 2)
for i in range(0, 1500, 100):
    cv2.line(img, (0, i), (2000, i), (255, 255, 255), 2)
    
cv2.circle(img, (1000, 750), 100, (255, 0, 0), -1)

cv2.imwrite('tests/temp_test/large.png', img)
print("Created tests/temp_test/large.png")
