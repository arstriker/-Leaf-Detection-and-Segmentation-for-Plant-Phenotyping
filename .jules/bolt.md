## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-20 - Inefficient Image Masking
**Learning:** Using `cv2.bitwise_and` to extract specific pixel colors from a masked region is highly inefficient because it allocates a full-size intermediate image Array where the vast majority of pixels are intentionally 0.
**Action:** When extracting properties of a masked region, use direct NumPy boolean indexing (e.g. `pixels = image[mask > 0]`) to fetch exactly the valid pixels in one step, bypassing the intermediate image generation.
