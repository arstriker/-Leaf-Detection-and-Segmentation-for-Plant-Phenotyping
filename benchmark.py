import time
import numpy as np
from preprocess import segment_leaf, extract_features, extract_edges_and_texture, grayscale_and_standardize

# Create a dummy image
dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

mask = segment_leaf(dummy_img)
gray, _ = grayscale_and_standardize(dummy_img)
_, lbp, _ = extract_edges_and_texture(gray)

# Baseline
start = time.time()
for _ in range(10):
    extract_features(dummy_img, mask)
baseline_time = (time.time() - start) / 10

print(f"Baseline: {baseline_time:.4f} seconds per call")
