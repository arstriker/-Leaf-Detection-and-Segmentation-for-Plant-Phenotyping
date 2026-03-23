## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2024-05-28 - Avoid fetching large JSON columns from SQLite when not needed
**Learning:** Fetching a table with a large JSON column (like `traits_json` in `phenotyping_report`) and then dropping it in pandas incurs significant memory overhead and disk I/O, negating some of the benefit of using `LIMIT`.
**Action:** When a large JSON column is not needed by the frontend (e.g. for summarizing recent records), use a `SELECT` query that explicitly specifies only the required columns instead of `SELECT *`.
