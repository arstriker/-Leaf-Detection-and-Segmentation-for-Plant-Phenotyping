## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Boolean Indexing vs cv2.bitwise_and
**Learning:** In `preprocess.py`, `cv2.bitwise_and` was used to mask an image before extracting color features. This allocates a full-size intermediate array with a black background, which is unnecessary and wastes memory and CPU cycles when we only need the pixel values within the mask.
**Action:** Use direct boolean indexing (e.g., `image[:,:,0][mask > 0]`) to extract pixel values instead of full array masking for feature extraction.
