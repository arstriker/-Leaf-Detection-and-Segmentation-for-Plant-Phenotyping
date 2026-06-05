## 2024-05-14 - Direct Boolean Indexing in NumPy/OpenCV
**Learning:** Using `cv2.bitwise_and` creates a full-size intermediate array before values are extracted. This leads to redundant memory allocation and I/O.
**Action:** Use direct boolean indexing `image[:, :, channel][mask > 0]` to directly fetch values into a flattened 1D array.

## 2024-05-14 - Avoiding PIL -> NumPy -> PIL Conversions
**Learning:** Both PyTorch's native inference logic and Ultralytics YOLOv8 gracefully handle PIL Images directly. The Streamlit frontend initially provides PIL Images, but passing `np.array(image)` forces the classifiers to convert it back to a PIL Image internally.
**Action:** Pass the original PIL Image directly to these models.

## 2024-05-14 - SQL SELECT Subset for Performance
**Learning:** Using `SELECT *` defaults to fetching the massive `traits_json` column into memory, which slows down queries significantly.
**Action:** Specifically limit database fetches to the required summary columns when `traits_json` is not needed by the UI.
