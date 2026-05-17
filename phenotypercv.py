import cv2
import numpy as np

def CLAHE_correct_rgb(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L channel of an image in Lab color space.
    """
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return image_clahe

def find_endpoints(input_img):
    """
    Finds endpoints in a skeletonized image using Hit-or-Miss transform.
    input_img: Binary image (0 or 255)
    Returns: Image with endpoints marked.
    """
    out = np.zeros_like(input_img)

    kernels = []
    # Kernel 0
    k0 = np.array([[-1, -1, -1],
                   [-1,  1, -1],
                   [ 0,  1,  0]], dtype=np.int32)
    kernels.append(k0)

    # Kernel 1
    k1 = np.array([[-1, -1, -1],
                   [-1,  1,  0],
                   [-1,  0,  1]], dtype=np.int32)
    kernels.append(k1)

    for base_kernel in kernels:
        curr_kernel = base_kernel
        for _ in range(4):
            # Rotate 90 degrees clockwise (k=-1 or k=3 for rot90)
            # In C++ rotate(cur_ker, temp_ker, ROTATE_90_CLOCKWISE)
            # numpy rot90 is counter-clockwise by default.
            # So to simulate CW rotation, we use k=-1 (or k=3).
            curr_kernel = np.rot90(curr_kernel, k=-1)

            # Apply Hit-or-Miss
            # OpenCV python expects kernel to be np.int8 or similar?
            # It handles -1 as background match, 1 as foreground match, 0 as don't care.
            hitmiss = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, curr_kernel.astype(np.int8))
            out = cv2.bitwise_or(out, hitmiss)

    return out

def find_branchpoints(input_img):
    """
    Finds branchpoints in a skeletonized image.
    input_img: Binary image.
    Returns: Image with branchpoints marked.
    """
    out = np.zeros_like(input_img)

    kernels = []
    # k0
    kernels.append(np.array([[-1,  1, -1],
                             [ 1,  1,  1],
                             [-1, -1, -1]], dtype=np.int32))
    # k1
    kernels.append(np.array([[ 1, -1,  1],
                             [-1,  1, -1],
                             [ 1, -1, -1]], dtype=np.int32))
    # k2
    kernels.append(np.array([[ 1, -1,  1],
                             [ 0,  1,  0],
                             [ 0,  1,  0]], dtype=np.int32))
    # k3
    kernels.append(np.array([[-1,  1, -1],
                             [ 1,  1,  0],
                             [-1,  0,  1]], dtype=np.int32))
    # k4
    kernels.append(np.array([[-1,  1, -1],
                             [ 1,  1,  1],
                             [-1,  1, -1]], dtype=np.int32))

    for base_kernel in kernels:
        curr_kernel = base_kernel
        for _ in range(4):
            curr_kernel = np.rot90(curr_kernel, k=-1)
            hitmiss = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, curr_kernel.astype(np.int8))
            out = cv2.bitwise_or(out, hitmiss)

    return out

def prune(input_img, size):
    """
    Prunes branches smaller than size.
    input_img: Skeletonized binary image.
    size: Number of iterations (pixels) to prune.
    """
    out = input_img.copy()
    for _ in range(size):
        endpoints = find_endpoints(out)
        out = cv2.subtract(out, endpoints)
    return out

def segment_skeleton(input_img):
    """
    Segments a skeleton by removing branchpoints.
    Returns: Image with segments separated (branchpoints removed).
    """
    bp = find_branchpoints(input_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bp_dil = cv2.dilate(bp, kernel, iterations=1)
    segments = cv2.subtract(input_img, bp_dil)
    return segments

def color_homography(img, r_coef, g_coef, b_coef):
    """
    Applies color homography using cubic polynomial coefficients.
    img: Input BGR image.
    r_coef, g_coef, b_coef: Coefficients arrays/lists of length 9.
        Order: b, g, r, b2, g2, r2, b3, g3, r3
    Returns: Corrected BGR image.
    """
    b, g, r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    b2 = b * b
    g2 = g * g
    r2 = r * r

    b3 = b2 * b
    g3 = g2 * g
    r3 = r2 * r

    # Helper to compute channel
    def compute_channel(coefs):
        # coefs order: b, g, r, b2, g2, r2, b3, g3, r3
        res = (b * coefs[0] + g * coefs[1] + r * coefs[2] +
               b2 * coefs[3] + g2 * coefs[4] + r2 * coefs[5] +
               b3 * coefs[6] + g3 * coefs[7] + r3 * coefs[8])
        return res

    new_b = compute_channel(b_coef)
    new_g = compute_channel(g_coef)
    new_r = compute_channel(r_coef)

    # Clip and convert back to uint8
    new_b = np.clip(new_b, 0, 255).astype(np.uint8)
    new_g = np.clip(new_g, 0, 255).astype(np.uint8)
    new_r = np.clip(new_r, 0, 255).astype(np.uint8)

    return cv2.merge((new_b, new_g, new_r))
