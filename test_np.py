import time
import numpy as np
import cv2

image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8) * 255

start = time.time()
for _ in range(100):
    valid_mask = mask > 0
    pixels = image[valid_mask]
    R_new = pixels[:, 0]
    G_new = pixels[:, 1]
    B_new = pixels[:, 2]
    mR = np.mean(R_new)
    mG = np.mean(G_new)
    mB = np.mean(B_new)
end1 = time.time()

start2 = time.time()
for _ in range(100):
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    R_old = img_masked[:, :, 0][mask > 0]
    G_old = img_masked[:, :, 1][mask > 0]
    B_old = img_masked[:, :, 2][mask > 0]
    mR = np.mean(R_old)
    mG = np.mean(G_old)
    mB = np.mean(B_old)
end2 = time.time()

print(f"Direct boolean indexing: {end1 - start:.4f}s")
print(f"cv2.bitwise_and: {end2 - start2:.4f}s")
