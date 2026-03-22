## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.

## 2025-02-12 - Database Fetch Optimization
**Learning:** In SQLite databases used by Streamlit, using `SELECT *` when large JSON or Blob columns exist (like `traits_json`) can drastically increase I/O reads and the memory required to instantiate Pandas DataFrames, even if those columns are immediately dropped by the frontend.
**Action:** Always write targeted SQL queries that explicitly SELECT only the columns necessary for the frontend display, preventing large unneeded columns from ever crossing the database boundary.
