import sys
from unittest.mock import MagicMock

# Mock out cv2, numpy, skimage, and others BEFORE importing preprocess
import builtins
class MockArray:
    def __init__(self, val=0):
        self.val = val
        self.shape = (10, 10, 3)

    def __gt__(self, other):
        return MockArray(1)

    def __lt__(self, other):
        return MockArray(0)

    def __eq__(self, other):
        return MockArray(1)

    def __getitem__(self, item):
        return MockArray(self.val)

    def astype(self, t):
        return MockArray(self.val)

    def __len__(self):
        return 1

    def copy(self):
        return MockArray(self.val)

    def __truediv__(self, other):
        return MockArray(self.val)

    def __mul__(self, other):
        return MockArray(self.val)

class NumpyMock:
    def __init__(self):
        self.uint8 = int
        self.float32 = float
        self.uint32 = int

    def mean(self, x):
        return 0.5

    def std(self, x):
        return 0.1

    def argmax(self, x):
        return 0

    def zeros_like(self, x):
        return MockArray(0)

class Cv2Mock:
    def __init__(self):
        self.COLOR_RGB2GRAY = 1
        self.NORM_MINMAX = 2
        self.CV_8U = 3
        self.THRESH_BINARY = 4
        self.THRESH_OTSU = 5
        self.MORPH_ELLIPSE = 6
        self.MORPH_OPEN = 7
        self.MORPH_CLOSE = 8
        self.CC_STAT_AREA = 9

    def cvtColor(self, a, b):
        return MockArray(0)

    def createCLAHE(self, clipLimit, tileGridSize):
        m = MagicMock()
        m.apply.return_value = MockArray(0)
        return m

    def normalize(self, a, b, c, d, e, dtype=None):
        return MockArray(0)

    def GaussianBlur(self, a, b, c):
        return MockArray(0)

    def Canny(self, a, b, c):
        return MockArray(0)

    def bitwise_and(self, a, b, mask=None):
        return MockArray(0)

class SkimageFeatureMock:
    def local_binary_pattern(self, a, b, c, method):
        return MockArray(0)

class SkimageMeasureMock:
    def regionprops(self, a):
        prop = MagicMock()
        prop.area = 100
        prop.perimeter = 50
        prop.eccentricity = 0.5
        prop.solidity = 0.9
        prop.extent = 0.8
        prop.major_axis_length = 20
        prop.minor_axis_length = 10
        return [prop]

    def label(self, a):
        return MockArray(0)

sys.modules['cv2'] = Cv2Mock()
sys.modules['numpy'] = NumpyMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.feature'] = SkimageFeatureMock()
sys.modules['skimage.measure'] = SkimageMeasureMock()
sys.modules['plantcv'] = MagicMock()
sys.modules['plantcv.plantcv'] = MagicMock()

# Now import
from preprocess import extract_features, grayscale_and_standardize, extract_edges_and_texture

mock_image = MockArray()
mock_mask = MockArray()
mock_lbp = MockArray()

def test_extract_features_no_lbp():
    features = extract_features(mock_image, mock_mask)
    assert 'lbp_mean' in features
    print("test_extract_features_no_lbp passed")

def test_extract_features_with_lbp():
    features = extract_features(mock_image, mock_mask, lbp=mock_lbp)
    assert 'lbp_mean' in features
    print("test_extract_features_with_lbp passed")

if __name__ == "__main__":
    test_extract_features_no_lbp()
    test_extract_features_with_lbp()
    print("All tests passed!")
