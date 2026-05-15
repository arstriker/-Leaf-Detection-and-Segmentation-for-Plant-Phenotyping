## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.
## 2025-02-12 - Redundant Array Allocation in Image Processing
**Learning:** In the `extract_features` pipeline, using `cv2.bitwise_and(image, image, mask=mask)` simply to extract specific valid pixels (e.g., color channels within a mask) is highly inefficient. It allocates a full `(H, W, 3)` intermediate array before slicing.
**Action:** Use direct NumPy boolean indexing (e.g., `valid_pixels = image[mask > 0]`) to efficiently extract only the relevant pixels without allocating full-size intermediate arrays.
