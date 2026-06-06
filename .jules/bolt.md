## 2025-02-12 - NumPy Boolean Indexing vs OpenCV Masking
**Learning:** In NumPy-based image processing pipelines, using OpenCV's `cv2.bitwise_and` followed by boolean channel masking to extract features allocates large $O(N)$ intermediate arrays unnecessarily. Direct boolean indexing `image[mask > 0]` is significantly faster and more memory-efficient.
**Action:** Always prefer direct boolean indexing over allocating full-sized intermediate masked arrays when extracting specific subsets of valid pixels for feature calculations.
