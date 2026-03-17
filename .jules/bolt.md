## 2026-03-17 - Optimize Database I/O Overhead
**Learning:** Returning large JSON columns using `SELECT *` for rendering lists on the frontend introduces significant memory and I/O overhead.
**Action:** Always specify explicit column names to exclude large textual/JSON fields when querying summaries or lists.
