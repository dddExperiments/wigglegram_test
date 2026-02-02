import numpy as np
import cv2
import os

os.makedirs('tests/temp_test', exist_ok=True)

def create_image(filename):
    # Create 800x600 image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Add patterns
    for i in range(0, 800, 50):
        cv2.line(img, (i, 0), (i, 600), (255, 255, 255), 1)
    for i in range(0, 600, 50):
        cv2.line(img, (0, i), (800, i), (255, 255, 255), 1)
        
    cv2.circle(img, (np.random.randint(100, 700), np.random.randint(100, 500)), 50, (255, 0, 0), -1)
    cv2.imwrite(f'tests/temp_test/{filename}', img)
    print(f"Created tests/temp_test/{filename}")

for i in range(5):
    create_image(f'img_{i}.png')
