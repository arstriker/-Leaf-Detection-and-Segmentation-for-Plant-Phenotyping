## 2024-05-24 - Avoid SQLite SELECT * on Large JSON Columns
**Learning:** This application stores heavy computed data (`traits_json`) inside an SQLite row. Using `SELECT *` simply to truncate that column later in pandas causes a severe and unnecessary I/O bottleneck by loading megabytes of unneeded JSON text from disk to memory for the dashboard preview.
**Action:** Always specifically query only the needed columns (`SELECT id, timestamp, disease_class, confidence, num_leaves_detected`) when hydrating summary views in Streamlit to bypass the expensive JSON deserialization overhead.
