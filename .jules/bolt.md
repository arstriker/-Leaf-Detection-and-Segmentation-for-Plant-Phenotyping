## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.
## 2025-02-12 - Inefficient Masked Array Allocation
**Learning:** In the `preprocess.py` module, extracting pixels corresponding to a mask was previously done by allocating a full-size intermediate array using `cv2.bitwise_and` followed by redundant boolean mask evaluations per channel. This creates unnecessary memory overhead.
**Action:** Use direct numpy boolean indexing (`image[mask > 0]`) to efficiently extract valid pixels without creating large temporary arrays.
