## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Avoid full-size intermediate arrays in boolean indexing
**Learning:** In `preprocess.py`, creating a full-size intermediate masked image using `cv2.bitwise_and(image, image, mask=mask)` before extracting specific color channels with boolean indexing (`img_masked[:,:,0][mask > 0]`) wastes both memory and CPU cycles. The intermediate array allocation is unnecessary.
**Action:** Use direct boolean indexing on the original image (e.g., `image[:,:,0][mask > 0]`) to extract masked values efficiently without creating large intermediate arrays.
