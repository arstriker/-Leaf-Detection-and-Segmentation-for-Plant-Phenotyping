## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Inefficient bitwise masking for feature extraction
**Learning:** Using `cv2.bitwise_and` to create a full-size intermediate array merely to extract a subset of pixels based on a mask is a codebase-specific performance anti-pattern. This wastes significant memory and CPU cycles when extracting color channels or local features.
**Action:** Always prefer direct boolean indexing (e.g., `image[mask > 0]`) to extract pixel values directly from the original image array without creating full-size intermediate arrays.
