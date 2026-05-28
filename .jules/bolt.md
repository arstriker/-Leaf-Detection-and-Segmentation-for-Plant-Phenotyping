## 2024-05-24 - Numpy Vectorization over OpenCV Arrays
**Learning:** Using `cv2.bitwise_and` followed by boolean indexing creates a full-size intermediate array with an unnecessary black background, doubling memory usage in Python memory space before the actual target pixels are extracted.
**Action:** When extracting pixels based on a mask for feature calculation (not display), bypass OpenCV masking and use direct NumPy boolean indexing on the original array (`image[mask > 0]`) to immediately isolate the target pixel values.
