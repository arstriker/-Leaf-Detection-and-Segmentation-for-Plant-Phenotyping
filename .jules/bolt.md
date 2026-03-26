## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-13 - Avoid SELECT * on large JSON columns
**Learning:** Using `SELECT *` on database tables with large JSON columns (like `traits_json` in `phenotyping_report`) unnecessarily loads massive strings into memory when fetching summary data for a UI, which then just gets dropped before rendering.
**Action:** Always use explicitly defined `SELECT` statements (e.g., `SELECT id, timestamp, disease_class`) instead of `SELECT *` when fetching list/summary records to minimize I/O and memory overhead.
