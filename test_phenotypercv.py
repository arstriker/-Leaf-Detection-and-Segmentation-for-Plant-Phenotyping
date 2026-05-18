import cv2
import numpy as np
import phenotypercv
import unittest

class TestPhenotyperCV(unittest.TestCase):
    def test_CLAHE_correct_rgb(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Gradient
        for i in range(100):
            img[i, :, 0] = i * 2
            img[i, :, 1] = i * 2
            img[i, :, 2] = i * 2

        out = phenotypercv.CLAHE_correct_rgb(img)
        self.assertEqual(out.shape, img.shape)
        # We expect some change
        self.assertFalse(np.array_equal(out, img))

    def test_find_endpoints(self):
        # Create a line
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 2:8] = 255 # Line from (2,5) to (7,5)

        # Endpoints should be at (2,5) and (7,5)
        endpoints = phenotypercv.find_endpoints(img)

        # Check locations
        self.assertEqual(endpoints[5, 2], 255)
        self.assertEqual(endpoints[5, 7], 255)
        # Should be only 2 pixels
        self.assertEqual(np.sum(endpoints > 0), 2)

    def test_find_branchpoints(self):
        # Create a cross
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 2:8] = 255 # Horizontal
        img[2:8, 5] = 255 # Vertical

        # Branchpoint at (5,5)
        bp = phenotypercv.find_branchpoints(img)

        self.assertEqual(bp[5, 5], 255)
        self.assertEqual(np.sum(bp > 0), 1)

    def test_segment_skeleton(self):
        # Create a T-shape
        img = np.zeros((10, 10), dtype=np.uint8)
        img[2:8, 5] = 255 # Vertical
        img[2, 2:8] = 255 # Horizontal top

        # Branchpoint at (2,5)
        segments = phenotypercv.segment_skeleton(img)

        # The branchpoint (2,5) and neighbors should be removed (due to dilate)
        # Original: (2,5) is intersection.
        # find_branchpoints returns (2,5).
        # dilate(bp) makes 3x3 square around (2,5) zero in segments.

        self.assertEqual(segments[2, 5], 0)
        self.assertEqual(segments[2, 2], 255) # tip of T
        self.assertEqual(segments[7, 5], 255) # bottom of T

if __name__ == '__main__':
    unittest.main()
