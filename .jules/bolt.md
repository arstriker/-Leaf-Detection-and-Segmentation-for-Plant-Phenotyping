## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - SQLite SELECT * Overhead
**Learning:** In `database.py`, using `SELECT *` to fetch all columns from `phenotyping_report` in `get_recent_reports()` caused the application to load the potentially large `traits_json` column into memory, even though the Streamlit UI immediately dropped it from the Pandas DataFrame because it wasn't needed for the summary display. This creates unnecessary disk I/O and memory usage.
**Action:** Always explicitly specify required columns in SQLite queries (`SELECT id, timestamp...`) to avoid fetching large JSON payloads or BLOBs when they are not needed for the current view.
