## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-13 - Avoid fetching large JSON payloads
**Learning:** In the `Automated Leaf Detection and Phenotyping` app's SQLite implementation, running `SELECT *` retrieves all columns, including potentially large TEXT blobs storing JSON representations (`traits_json`). When populating a Streamlit summary dashboard, these columns are never used but still allocate significant I/O time and memory overhead.
**Action:** Always explicitly select only the required summary columns when querying records destined for partial-display tables, avoiding the implicit fetch of large, unused data payloads.
