## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Array Boolean Indexing Overhead
**Learning:** Using `cv2.bitwise_and(image, image, mask=mask)` allocates a full WxHxC intermediate array. If the only subsequent operation is to extract values using boolean indexing (e.g. `R = img_masked[:,:,0][mask > 0]`), the `bitwise_and` is completely redundant. Applying the boolean mask directly to the original array (e.g. `R = image[:,:,0][mask > 0]`) eliminates the large memory allocation and is significantly faster.
**Action:** When extracting values from an array using a mask, apply boolean indexing directly to the source array instead of creating an intermediate masked copy.
