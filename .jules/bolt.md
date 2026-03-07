## 2024-03-07 - Database Fetch & PIL conversion optimizations
**Learning:** Streamlit data structures can load significantly faster when avoiding fetching unneeded large JSON columns by specifying the SQL columns directly instead of SELECT *. Reusing PIL images in YOLOv8 & PyTorch classifiers saves memory overhead compared to repeated implicit NumPy conversions.
**Action:** Next time working with image processing pipelines and Streamlit databases, audit variable typing to prefer native PIL conversions and ensure SQL selects avoid hidden large JSON columns.
