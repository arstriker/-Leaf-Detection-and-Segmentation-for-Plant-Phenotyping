## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - SQLite `SELECT *` with Large JSON Columns
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, fetching recent reports from the SQLite database using `SELECT *` was loading a large `traits_json` column into memory for every record, only to be immediately dropped in the frontend by Pandas before rendering the Streamlit UI. This caused unnecessary memory overhead and database I/O.
**Action:** Always optimize database queries by explicitly selecting only the required columns, especially when dealing with large text or JSON payloads that are not needed for the immediate view.
