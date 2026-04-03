## 2024-04-03 - [SQLite DB Optimization]
**Learning:** Found that the traits_json column in the SQLite DB can get large, but the UI doesn't use it in `get_recent_reports`.
**Action:** When creating a subset fetch for large datasets in SQLite, modify the SELECT query to explicitly skip large columns (like JSON blobs) instead of `SELECT *` to reduce I/O and memory overhead.
