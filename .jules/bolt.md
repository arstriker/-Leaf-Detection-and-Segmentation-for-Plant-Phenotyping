## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-04-19 - Redundant Full-Size Array Allocation for Masked Extraction
**Learning:** In NumPy/OpenCV-based image processing, using `cv2.bitwise_and(image, image, mask=mask)` to extract features from a masked region allocates a completely new, full-sized image array in memory. When the goal is simply to compute statistics (like mean/std) over the non-zero mask pixels, this is highly inefficient.
**Action:** Replace `cv2.bitwise_and` feature extraction with direct multidimensional boolean indexing (e.g., `image[:,:,0][mask > 0]`) to completely avoid the intermediate O(W*H*C) array allocation and save significant memory and CPU cycles.
