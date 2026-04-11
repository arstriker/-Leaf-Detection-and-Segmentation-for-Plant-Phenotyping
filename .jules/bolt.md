## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2026-04-11 - Intermediate Array Allocation Avoidance
**Learning:** In NumPy/OpenCV workflows, using functions like `cv2.bitwise_and` to mask images creates full-size intermediate arrays in memory. If you only intend to process the masked pixels afterwards, direct boolean indexing (e.g., `image[mask > 0]`) bypasses this O(N) allocation entirely.
**Action:** Replace `cv2.bitwise_and` with direct boolean indexing when extracting specific feature channels or pixels based on a binary mask to reduce memory overhead and CPU cycles.