import cv2
import numpy as np
import time

def original_method(image, mask):
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    R = img_masked[:, :, 0][mask > 0]
    G = img_masked[:, :, 1][mask > 0]
    B = img_masked[:, :, 2][mask > 0]
    return R, G, B

def optimized_method(image, mask):
    valid_mask = mask > 0
    pixels = image[valid_mask]
    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]
    return R, G, B

# Setup
image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
mask = np.random.choice([0, 255], (1000, 1000), p=[0.7, 0.3]).astype(np.uint8)

# Warmup
original_method(image, mask)
optimized_method(image, mask)

n_iters = 100

t0 = time.time()
for _ in range(n_iters):
    original_method(image, mask)
t1 = time.time()
print(f"Original: {(t1-t0)/n_iters:.6f} seconds per iteration")

t0 = time.time()
for _ in range(n_iters):
    optimized_method(image, mask)
t1 = time.time()
print(f"Optimized: {(t1-t0)/n_iters:.6f} seconds per iteration")
