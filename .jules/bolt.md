## 2024-05-24 - PIL vs NumPy Conversion Overhead in PyTorch/YOLO
**Learning:** PyTorch/Ultralytics models natively accept PIL Images and implicitly convert NumPy arrays back to PIL if provided. In `app.py`, passing `img_np` to `classifier.classify()` forced a redundant `Image.fromarray(image)` conversion on every inference, wasting memory and CPU cycles.
**Action:** When working with Streamlit + PyTorch/YOLO, always preserve the initial PIL Image object and pass it directly to classification methods to avoid double-conversion overhead.

## 2024-05-24 - SQLite JSON Column Memory Bloat in Streamlit
**Learning:** In `database.py`, using `SELECT *` to fetch the top 5 recent records also pulled in the `traits_json` column (a large JSON payload) for every row, even though the frontend immediately discarded it (`.drop(columns=["Traits JSON"])`).
**Action:** Always explicitly specify required columns in SQLite queries (`SELECT id, timestamp...`) when populating frontend summary tables to drastically reduce I/O and memory usage.
