import sys
from unittest.mock import MagicMock, patch

# Mocking external dependencies
try:
    import numpy as np
except ImportError:
    # Create a more capable mock for numpy
    mock_np = MagicMock()
    sys.modules["numpy"] = mock_np
    np = mock_np

    # Help the mock numpy behave a bit more like the real thing for these tests
    def mock_zeros(shape, **kwargs):
        m = MagicMock()
        m.__gt__.return_value = m
        m.astype.return_value = m
        m.__getitem__.return_value = m
        return m

    def mock_ones(shape, **kwargs):
        m = MagicMock()
        m.__gt__.return_value = m
        m.astype.return_value = m
        m.__getitem__.return_value = m
        return m

    np.zeros.side_effect = mock_zeros
    np.ones.side_effect = mock_ones
    np.mean.return_value = 0.0
    np.std.return_value = 0.0

mock_cv2 = MagicMock()
mock_skimage = MagicMock()

sys.modules["cv2"] = mock_cv2
sys.modules["skimage"] = mock_skimage
sys.modules["skimage.feature"] = mock_skimage.feature
sys.modules["skimage.measure"] = mock_skimage.measure

# Import the module to test
sys.modules["plantcv"] = MagicMock()
sys.modules["plantcv.plantcv"] = MagicMock()
import preprocess


def test_extract_features_empty_mask():
    # Setup: Create a dummy image and an empty mask
    image = np.zeros((100, 100, 3))
    mask = np.zeros((100, 100))

    # Mock label and regionprops to return no properties (empty list)
    with patch("preprocess.label", return_value=np.zeros((100, 100))), patch(
        "preprocess.regionprops", return_value=[]
    ):

        result = preprocess.extract_features(image, mask)

    # Assert
    assert result == {"error": "No leaf found in mask."}


def test_extract_features_valid_mask():
    # Setup: Create a dummy image and a mask with one "leaf"
    image = np.zeros((100, 100, 3))
    mask = np.ones((100, 100))

    # Mock a region property object
    mock_prop = MagicMock()
    mock_prop.area = 100
    mock_prop.perimeter = 40
    mock_prop.eccentricity = 0.5
    mock_prop.solidity = 0.9
    mock_prop.extent = 0.8
    mock_prop.axis_major_length = 15
    mock_prop.axis_minor_length = 10

    # Mock the return values for dependencies
    with patch("preprocess.label", return_value=np.ones((100, 100))), patch(
        "preprocess.regionprops", return_value=[mock_prop]
    ), patch("preprocess.cv2.bitwise_and", return_value=image), patch(
        "preprocess.grayscale_and_standardize",
        return_value=(np.zeros((100, 100)), np.zeros((100, 100))),
    ), patch(
        "preprocess.extract_edges_and_texture",
        return_value=(None, np.zeros((100, 100)), None),
    ):

        # In extract_features, there is: R = img_masked[:,:,0][mask > 0]
        # and: if len(R) > 0:
        # If R is a mock, len(R) might fail or return something unexpected.
        # We want len(R) > 0 to test the color feature extraction path.

        mock_R = MagicMock()
        mock_R.__len__.return_value = 100

        # This is getting complicated because of the chained indexing: img_masked[:,:,0][mask > 0]
        image.__getitem__.return_value.__getitem__.return_value = mock_R

        result = preprocess.extract_features(image, mask)

    # Assertions for some key features
    assert "area_px" in result
    assert result["area_px"] == 100
    assert "perimeter_px" in result
    assert result["perimeter_px"] == 40
    assert "aspect_ratio" in result
    assert abs(result["aspect_ratio"] - 1.5) < 1e-5
