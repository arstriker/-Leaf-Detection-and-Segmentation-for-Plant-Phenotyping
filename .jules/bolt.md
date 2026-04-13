## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.
## 2025-02-12 - Redundant Image Masking Allocation
**Learning:** Using `cv2.bitwise_and` to apply a mask before extracting pixels (e.g., for color features) unnecessarily allocates a full-size intermediate image array.
**Action:** When extracting specific pixels based on a mask, use direct boolean indexing (e.g., `image[:,:,0][mask > 0]`) to skip the full-array allocation step, saving memory and CPU cycles.
