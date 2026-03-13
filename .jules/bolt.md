## 2024-05-24 - Avoid SELECT * on large JSON columns
**Learning:** Selecting all columns (`SELECT *`) on SQLite tables containing large JSON text fields (e.g., `traits_json`) causes significant I/O and memory overhead, especially when those JSON fields aren't even used by the frontend (like in recent reports summaries).
**Action:** Always explicitly specify required columns in SQL queries to avoid unnecessary data loading, particularly for large payloads or summary views.
