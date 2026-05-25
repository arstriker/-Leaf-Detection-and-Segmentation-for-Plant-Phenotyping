1. *Clean up syntax errors in `preprocess.py`*
   - Fix duplicate function definitions for `segment_leaf` and `extract_features`.
   - Remove duplicate lines caused by merge conflicts.
2. *Implement Bolt performance optimization in `preprocess.py`*
   - Replace `cv2.bitwise_and(image, image, mask=mask)` with direct boolean indexing `mask_bool = mask > 0; R = image[:,:,0][mask_bool]` in `extract_features` to avoid full-size array allocation.
   - Add comments explaining the optimization.
3. *Verify functionality locally*
   - Use a script to test `extract_features` to ensure it still works correctly.
   - Run `python3 -m py_compile preprocess.py` to check for syntax errors.
4. *Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.*
   - Follow instructions from the pre commit tool.
5. *Submit PR*
   - Branch: `bolt-opt-numpy-indexing`
   - Title: `⚡ Bolt: Use direct boolean indexing instead of cv2.bitwise_and`
   - Description must contain `💡 What`, `🎯 Why`, `📊 Impact`, and `🔬 Measurement`.
