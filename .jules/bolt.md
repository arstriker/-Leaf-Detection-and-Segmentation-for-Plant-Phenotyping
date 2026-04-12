## 2024-05-23 - Optimizing NumPy Masking
**Learning:** Using `cv2.bitwise_and` followed by boolean indexing (`mask > 0`) is inefficient because it creates a full-sized intermediate array (masked image). A more optimal way to extract pixels within a mask is to directly index the image array with the boolean mask: `pixels = image[mask > 0]`.
**Action:** Use direct boolean indexing `image[mask > 0]` to extract pixels instead of `cv2.bitwise_and` when only the masked pixels are needed (e.g. for calculating mean/std).
