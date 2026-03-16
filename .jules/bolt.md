## 2024-05-24 - [Avoid PIL to NumPy double conversions]
**Learning:** PyTorch models explicitly expect PIL Image inputs. Passing NumPy arrays from frontend results in redundant back-and-forth conversions, adding significant memory overhead and consuming extra CPU cycles. Furthermore, YOLOv8 gracefully handles PIL directly without user-side intervention.
**Action:** When working on frontend UI layers that perform inferences, pass PIL image references directly instead of standardizing into NumPy early in the pipeline if it isn't strictly required.
