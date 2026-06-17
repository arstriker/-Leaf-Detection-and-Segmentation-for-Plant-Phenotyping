## 2024-06-17 - Redundant Mask Evaluation in NumPy
**Learning:** Using `cv2.bitwise_and` to extract color channel pixels based on a binary mask in NumPy arrays results in significant overhead by allocating a new image array, followed by redundant `mask > 0` evaluations for each channel.
**Action:** Replace `cv2.bitwise_and` masking with direct boolean array indexing (e.g., `valid_mask = mask > 0; R = image[:,:,0][valid_mask]`) to optimize both CPU cycles and memory footprint when only valid pixels are needed.
