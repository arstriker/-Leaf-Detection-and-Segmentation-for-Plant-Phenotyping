import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label
from skimage.measure import regionprops, label
from plantcv import plantcv as pcv

# Configure PlantCV globally
pcv.params.debug = None

# Try to import phenotypercv if it exists
try:
    import phenotypercv

    PHENOTYPER_CV_AVAILABLE = True
except ImportError:
    PHENOTYPER_CV_AVAILABLE = False


def grayscale_and_standardize(image):
    """
    Converts RGB image to grayscale and standardizes the pixel values.
    Returns the grayscale image and a standardized float image [0, 1].
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    std_img = gray.astype(np.float32) / 255.0
    return gray, std_img


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)

def apply_color_clahe(image):
    """
    Applies CLAHE to the L channel of the image (Lab color space).
    Uses phenotypercv if available, otherwise falls back to OpenCV implementation.
    """
    if PHENOTYPER_CV_AVAILABLE:
        return phenotypercv.CLAHE_correct_rgb(image)

    # Fallback implementation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def extract_color_indices(image):
    """
    Extracts Excess Green (ExG) and Excess Red (ExR) indices.
    Image should be RGB uint8.
    """
    # Convert to float to avoid overflow
    img_float = image.astype(np.float32)
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    # Normalize by total sum of R, G, B
    total = R + G + B
    total[total == 0] = 1  # avoid division by zero

    r = R / total
    g = G / total
    b = B / total

    # ExG = 2g - r - b
    exg = 2 * g - r - b

    # ExR = 1.4r - g
    exr = 1.4 * r - g

    # Normalize to [0, 255] for visualization
    exg_vis = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    exr_vis = cv2.normalize(exr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return exg, exr, exg_vis, exr_vis


def extract_edges_and_texture(gray_image):
    """
    Extracts edges using Canny and texture using Local Binary Patterns (LBP).
    """
    # Canny Edge Detection
    # Using automatic optimal thresholding (Otsu) to find bounds could be better,
    # but fixed bounds with a slight blur works well for leaves.
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Local Binary Patterns
    # Settings for LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")

    # Convert LBP to visualizable format
    lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return edges, lbp, lbp_vis


def segment_leaf(image, exg=None):
    """
    Segments the leaf from the background using PlantCV.
    Returns a binary mask (0 for background, 255 for leaf).
    """
    if PHENOTYPER_CV_AVAILABLE:
        # Fallback to custom logic if phenotypercv API is unknown.
        # Assuming there is some segment function: phenotypercv.segment(image)
        # For now, if someone installs it but we don't know the API, handle it gracefully.
        pass

    # Standard Computer Vision Fallback
    # 1. Convert to Lab color space. Leaf is usually very pronounced in 'a' and 'b' channels.
    # Alternatively, use ExG which is robust for green leaves.
    exg, exr, exg_vis, exr_vis = extract_color_indices(image)

    if exg is None:
        exg, exr, exg_vis, exr_vis = extract_color_indices(image)
    
    # Threshold ExG using Otsu's method
    # Need to convert ExG to uint8 properly mapped to [0, 255]
    exg_mapped = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu thresholding
    _, mask = cv2.threshold(exg_mapped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Refine mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (assuming the leaf is the largest object)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

#    if num_labels > 1:
#        # sizes are in the last column of stats
#        # The 0th label is the background. Extract the max size among the others.
    # 1. Convert to a colorspace that isolates green (e.g. LAB)
    a_channel = pcv.rgb2gray_lab(rgb_img=image, channel='a')

    # 2. Threshold the 'a' channel to separate leaf from background
    # Green plants appear dark in the 'a' channel.
    # We use an inverted auto threshold to get a white leaf on a black background
    thresh = pcv.threshold.otsu(gray_img=a_channel, object_type='dark')

    # 3. Clean up the mask
    # Fills small holes within the leaf
    mask = pcv.fill(bin_img=thresh, size=50)
    
    # Optional: clean up noise in the background
    # mask = pcv.fill_holes(bin_img=mask)
    
    # 4. Find connected components (objects)
    # This acts like finding the largest contours
    labeled_mask, num_objects = pcv.create_labels(mask=mask)
        
    # Isolate the largest object
    # If there are multiple parts we want the main leaf
    if num_objects > 1:
        # pcv.roi.multi doesn't easily return the largest by default
        # But we can use standard OpenCV to pull out the largest blob 
        # from PlantCV's clean mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.zeros_like(mask)
        final_mask[labels == largest_label] = 255
    else:
        final_mask = mask

    return final_mask


def extract_features(image, mask, gray_image=None, lbp=None):
    """
    Extracts key phenotypic traits from the segmented leaf.
    Includes Shape, Size, Texture, Color.
    Returns a dictionary of traits.
    """
    features = {}

    # Ensure mask is binary [0, 1] for regionprops
    binary_mask = (mask > 0).astype(int)

    # --- Shape and Size Features ---
    # label the image (though there should be only 1 leaf)
    lbl_img = label(binary_mask)
    props = regionprops(lbl_img)

    if len(props) == 0:
        return {"error": "No leaf found in mask."}

    leaf_prop = props[0]  # assuming largest/only object

    features["area_px"] = leaf_prop.area
    features["perimeter_px"] = leaf_prop.perimeter
    features["eccentricity"] = leaf_prop.eccentricity
    features["solidity"] = leaf_prop.solidity
    features["extent"] = leaf_prop.extent
    features["aspect_ratio"] = leaf_prop.major_axis_length / (
        leaf_prop.minor_axis_length + 1e-6
    )

        
    leaf_prop = props[0] # assuming largest/only object
    
    features['area_px'] = leaf_prop.area
    features['perimeter_px'] = leaf_prop.perimeter
    features['eccentricity'] = leaf_prop.eccentricity
    features['solidity'] = leaf_prop.solidity
    features['extent'] = leaf_prop.extent
    features['aspect_ratio'] = leaf_prop.axis_major_length / (leaf_prop.axis_minor_length + 1e-6)
    
    # --- Color Features (only within the mask) ---
    # ⚡ Bolt Optimization: Use direct boolean indexing to avoid allocating
    # a full-size intermediate array with cv2.bitwise_and.
    # This reduces memory allocation and saves CPU cycles.
    valid_mask = mask > 0
    R = image[:, :, 0][valid_mask]
    G = image[:, :, 1][valid_mask]
    B = image[:, :, 2][valid_mask]

    if len(R) > 0:
        features["mean_R"] = float(np.mean(R))
        features["mean_G"] = float(np.mean(G))
        features["mean_B"] = float(np.mean(B))
        features["std_R"] = float(np.std(R))
        features["std_G"] = float(np.std(G))
        features["std_B"] = float(np.std(B))
    else:
        features["mean_R"] = features["mean_G"] = features["mean_B"] = 0.0
        features["std_R"] = features["std_G"] = features["std_B"] = 0.0

    # --- Texture Features (LBP) ---
    if lbp is None:
        gray_image, _ = grayscale_and_standardize(image)
        _, lbp, _ = extract_edges_and_texture(gray_image)
    if gray_image is None or lbp is None:
        if gray_image is None:
            gray_image, _ = grayscale_and_standardize(image)
        if lbp is None:
            _, lbp, _ = extract_edges_and_texture(gray_image)

    lbp_masked = lbp[mask > 0]

    if len(lbp_masked) > 0:
        features["lbp_mean"] = float(np.mean(lbp_masked))
        features["lbp_std"] = float(np.std(lbp_masked))
    else:
        features["lbp_mean"] = features["lbp_std"] = 0.0

        features['lbp_mean'] = features['lbp_std'] = 0.0

    # --- Skeleton Analysis (PhenotyperCV) ---
    if PHENOTYPER_CV_AVAILABLE:
        try:
            from skimage.morphology import skeletonize
            skel_bool = skeletonize(binary_mask > 0)
            skel = (skel_bool * 255).astype(np.uint8)

            endpoints = phenotypercv.find_endpoints(skel)
            branchpoints = phenotypercv.find_branchpoints(skel)

            features['num_endpoints'] = int(np.sum(endpoints > 0))
            features['num_branchpoints'] = int(np.sum(branchpoints > 0))
        except ImportError:
            pass
        
    return features


if __name__ == "__main__":
    # Simple test for preprocessing module
    print("Testing Preprocessing Module...")

    # Create a dummy image (RGB)
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Apply processing
    gray, std = grayscale_and_standardize(dummy_img)
    clahe = apply_clahe(gray)
    exg, exr, exg_vis, exr_vis = extract_color_indices(dummy_img)
    edges, lbp, lbp_vis = extract_edges_and_texture(gray)
    mask = segment_leaf(dummy_img)
    features = extract_features(dummy_img, mask, lbp=lbp)

    print("Testing successful. Output Shapes:")
    print(f"Gray: {gray.shape}, CLAHE: {clahe.shape}")
    print(f"ExG Vis: {exg_vis.shape}")
    print(f"Edges: {edges.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Features: {list(features.keys())[:5]} ...")
