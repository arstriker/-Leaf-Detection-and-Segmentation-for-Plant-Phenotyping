
## 2024-05-15 - Minimize PIL-to-NumPy conversion
**Learning:** Passing PIL images directly to PyTorch/YOLO classifiers avoids redundant numpy-to-PIL conversion overhead. PyTorch models trained on torchvision typically expect PIL images natively or are easily converted to Tensors without intermediate NumPy conversion if we provide PIL. YOLOv8 also gracefully accepts PIL images natively.
**Action:** When a PIL image is already loaded from disk/upload, pass the original PIL image directly into the classifiers instead of passing a numpy array that the classifier must immediately convert back to PIL internally.

## 2024-05-15 - Optimize Database Query Payload
**Learning:** Using `SELECT *` on tables with large JSON columns (like `traits_json`) to display summary tables in Streamlit causes unnecessary I/O and memory overhead when that column is immediately dropped on the frontend.
**Action:** Always selectively query only the columns needed by the UI instead of defaulting to `SELECT *`, especially when dealing with JSON payload columns in SQLite.
