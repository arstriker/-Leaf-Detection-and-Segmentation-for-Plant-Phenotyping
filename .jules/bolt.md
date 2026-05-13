## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.
## 2025-02-12 - Memory allocation in masked array operations
**Learning:** Using OpenCV's `bitwise_and` to apply a mask to an image creates a full-sized intermediate array (e.g. $H \times W \times 3$), which wastes memory if the operation only needs to extract pixel values within the mask.
**Action:** When extracting values bounded by a binary mask (e.g., getting all R, G, B values for masked pixels to compute statistical features), use direct boolean indexing (like `image[:,:,0][mask > 0]`) rather than allocating a full array with `cv2.bitwise_and`.
