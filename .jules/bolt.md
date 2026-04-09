## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Direct Boolean Indexing vs Bitwise And
**Learning:** In NumPy-based image processing pipelines, using `cv2.bitwise_and` to extract regions allocates a full-size intermediate array. If only a small subset of values is needed (e.g., within a segmented mask), direct boolean indexing (`image[mask > 0]`) is vastly more memory-efficient and requires fewer CPU cycles.
**Action:** Replace full-array bitwise operations with direct boolean indexing when only specific values (like color channels within a mask) are required for statistical extraction.
