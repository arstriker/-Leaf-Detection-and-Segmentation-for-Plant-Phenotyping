## 2025-02-12 - Redundant Image Conversions
**Learning:** In the `Automated Leaf Detection and Phenotyping` app, PIL images were being converted to NumPy arrays for some tasks, and then those NumPy arrays were being passed into PyTorch and YOLOv8 models. Both PyTorch and YOLOv8 natively handle PIL images and implicitly convert NumPy arrays back to PIL format internally, creating a redundant double-conversion overhead that wastes CPU cycles and memory.
**Action:** Always ensure that PIL images are preserved and passed directly to deep learning models like YOLOv8 and PyTorch to avoid unnecessary double-conversions when possible.
## 2025-02-12 - Reverting Bad Optims and Safety First
**Learning:** A seemingly straightforward removal of a redundant database connection in `app.py` caused significant instability and potential NameErrors due to scope variables, making the PR invalid.
**Action:** Always constrain the performance fixes to *exactly one* safe, measurable optimization per PR, and don't try to bundle other fixes together if it introduces multiple potential runtime crashes. Keep risk low and avoid unverified assumptions.
