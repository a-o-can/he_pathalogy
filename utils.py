import numpy as np
import math
import matplotlib.pyplot as plt


def _ensure_uint8(img, SCALE_255=False):
    if img.dtype == np.uint8:
        return img
    if SCALE_255 or (img.dtype in (np.float32, np.float64) and np.nanmax(img) <= 1.0 + 1e-6):
        img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8, copy=False)

def _compute_tile_scores(tiles_uint8):
    """
    Compute tissue presence scores for each tile using pure NumPy.

    Returns dict with arrays of shape (N,):
      - nonwhite_ratio
      - tissue_ratio
      - local_contrast
      - mean_intensity
    """
    if tiles_uint8.ndim != 4 or tiles_uint8.shape[-1] != 3:
        raise ValueError(f"Expecting (N, H, W, 3), got {tiles_uint8.shape}")

    N, H, W, _ = tiles_uint8.shape
    tiles_f = tiles_uint8.astype(np.float32, copy=False)

    R = tiles_f[..., 0]
    G = tiles_f[..., 1]
    B = tiles_f[..., 2]

    # Heuristic 1: nonwhite ratio
    nonwhite = np.any(tiles_uint8 < 240, axis=-1)                 # (N, H, W)
    nonwhite_ratio = nonwhite.reshape(N, -1).mean(axis=1)         # (N,)

    # Heuristic 2: tissue ratio using HSV S and V (computed in NumPy)
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    V = maxc                                                          # 0..255
    # S = (max - min)/max * 255, with S=0 when V==0
    denom = np.where(V == 0, 1.0, V)
    S = (maxc - minc) / denom * 255.0

    s_thresh = 15.0
    v_hi     = 250.0
    tissue_mask = (S >= s_thresh) & (V <= v_hi)
    tissue_ratio = tissue_mask.reshape(N, -1).mean(axis=1)

    # Heuristic 3: local contrast via grayscale std (ITU-R BT.601 luma)
    gray = (0.2989 * R + 0.5870 * G + 0.1140 * B)                  # (N, H, W)
    local_contrast = gray.reshape(N, -1).std(axis=1)
    mean_intensity = gray.reshape(N, -1).mean(axis=1)

    return {
        "nonwhite_ratio": nonwhite_ratio,
        "tissue_ratio": tissue_ratio,
        "local_contrast": local_contrast,
        "mean_intensity": mean_intensity,
    }

def find_full_tiles(
    tiles,
    SCALE_255=False,
    min_nonwhite_ratio=0.02,
    min_tissue_ratio=0.02,
    min_contrast=3.0,
):
    """
    Returns indices for tiles that likely contain tissue.

    Parameters
    ----------
    tiles : np.ndarray, shape (N, H, W, 3)
    SCALE_255 : bool
        If tiles are floats in [0, 1], set True to scale to [0, 255].
    min_nonwhite_ratio : float
        Minimum fraction of nonwhite pixels to keep a tile.
    min_tissue_ratio : float
        Minimum fraction of saturated pixels to keep a tile.
    min_contrast : float
        Minimum grayscale std per tile.

    Returns
    -------
    full_idx : np.ndarray of ints
        Indices of full tiles.
    scores : dict of np.ndarray
        Per tile metrics: nonwhite_ratio, tissue_ratio, local_contrast, mean_intensity.
    """
    tiles_u8 = _ensure_uint8(np.asarray(tiles), SCALE_255=SCALE_255)
    scores = _compute_tile_scores(tiles_u8)

    is_full = (
        (scores["nonwhite_ratio"] >= min_nonwhite_ratio) &
        (scores["tissue_ratio"]   >= min_tissue_ratio) &
        (scores["local_contrast"] >= min_contrast)
    )

    full_idx = np.flatnonzero(is_full)

    # Sort by tissue richness (tissue_ratio) for convenience
    order = np.argsort(-scores["tissue_ratio"][full_idx])
    full_idx = full_idx[order]

    print(f"Tiles total: {tiles.shape[0]}")
    print(f"Full tiles: {len(full_idx)}")
    return full_idx, scores

def plot_top_tiles(tiles, rank_scores, top_k=16, tile_titles=None):
    """
    Safe version: gracefully handles empty inputs (k == 0).
    """
    tiles = np.asarray(tiles)
    rank_scores = np.asarray(rank_scores)

    # Nothing to plot if no tiles or no scores
    if tiles.size == 0 or rank_scores.size == 0:
        print("No tiles to plot (received empty selection).")
        return

    N = tiles.shape[0]
    k = min(top_k, N)
    if k == 0:
        print("No tiles to plot (k == 0).")
        return

    order = np.argsort(-rank_scores)[:k]

    # Guard against degenerate layout (k could be 1)
    cols = max(1, int(math.ceil(math.sqrt(k))))
    rows = max(1, int(math.ceil(k / cols)))

    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i, idx in enumerate(order):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(tiles[idx].astype(np.uint8))
        title = f"idx {idx} | score {rank_scores[idx]:.3f}"
        if tile_titles is not None:
            title = f"{title}\n{tile_titles[idx]}"
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
"""
Slide-specific H&E deconvolution + nuclei segmentation (no OpenCV; works on your tiles).

Why your first try looked bad:
- Using a fixed/canonical stain matrix often fails when staining or scanning varies.
- Better: estimate the stain matrix per slide (Macenko), then deconvolve, then segment nuclei from the H (hematoxylin) OD channel.

What this cell does:
1) Robust per-tile tissue mask (to ignore background while estimating stains).
2) Estimate a 2-stain Macenko matrix W (H & E) from tissue pixels.
3) Deconvolve to get hematoxylin optical-density (H_OD).
4) Segment nuclei from H_OD with Otsu threshold + cleanup + watershed split.
5) Returns masks and shows a quick overlay.

Inputs expected:
- `image`: your uint8 array with shape (N, 256, 256, 3) as prepared earlier.

Dependencies (install if missing):
  pip install scikit-image scipy numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature, exposure, util, color

# ---------- Utilities ----------

def rgb_to_od(rgb_u8, I0=255.0):
    """Convert RGB uint8 (H,W,3) to optical density (H,W,3)."""
    # Add 1 to avoid log(0)
    rgb = rgb_u8.astype(np.float32)
    return -np.log((rgb + 1.0) / (I0 + 1.0))

def macenko_stain_matrix(rgb_u8, tissue_mask=None, I0=255.0, beta=0.15, alpha=1.0):
    """
    Estimate slide-specific 2-stain (H&E) matrix using Macenko.
    Returns W (3x3) with two columns for H & E and a third dummy column.

    Parameters
    ----------
    rgb_u8 : uint8 (H,W,3)
    tissue_mask : optional boolean (H,W) — restricts estimation to tissue pixels
    I0 : float — illumination white point (255 for uint8)
    beta : float — OD threshold to filter very light pixels (typ. 0.15)
    alpha : float — percentile for angle extremes (1 to 5 is common)

    Returns
    -------
    W : np.ndarray shape (3,3)   columns ~ [H, E, filler]
    """
    od = rgb_to_od(rgb_u8, I0=I0).reshape(-1, 3)  # (P,3)
    if tissue_mask is not None:
        od = od[tissue_mask.flatten()]

    # Filter pixels with low OD (background)
    od = od[np.all(od > beta, axis=1)]
    if od.size == 0:
        # Fallback to canonical H&E vectors (Ruifrok)
        W = np.array([[0.65, 0.07, 0.0],
                      [0.70, 0.99, 0.0],
                      [0.29, 0.11, 1.0]], dtype=np.float32)
        return W

    # SVD on centered OD
    od_centered = od - od.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(od_centered, full_matrices=False)
    # Top 2 PCs define the stain plane
    V = Vt[:2, :]  # (2,3)

    # Project OD onto plane and compute angles
    proj = od_centered @ V.T  # (P,2)
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # Robust extremes
    lo, hi = np.percentile(angles, [alpha, 100 - alpha])
    v1 = (V.T @ np.array([np.cos(lo), np.sin(lo)])).ravel()
    v2 = (V.T @ np.array([np.cos(hi), np.sin(hi)])).ravel()

    # Normalize stain vectors to unit length and ensure positive entries
    def _norm(v):
        v = v / (np.linalg.norm(v) + 1e-8)
        # enforce positive by flipping if needed (convention)
        if v[0] < 0: v = -v
        return v

    h_vec = _norm(v1)
    e_vec = _norm(v2)

    # Order heuristically so first is hematoxylin (higher blue component)
    if h_vec[2] < e_vec[2]:
        h_vec, e_vec = e_vec, h_vec

    # Third column as a filler to make 3x3 invertible
    # Use cross-product to span RGB space
    third = np.cross(h_vec, e_vec)
    if np.linalg.norm(third) < 1e-6:
        third = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    third = third / (np.linalg.norm(third) + 1e-8)

    W = np.stack([h_vec, e_vec, third], axis=1).astype(np.float32)  # (3,3)
    return W

def separate_stains_od(rgb_u8, W, I0=255.0):
    """
    Deconvolve: OD = M @ C  ->  C = M^{-1} @ OD
    Returns channels (H, E, third) in OD, each (H,W).
    """
    H, Wd, _ = rgb_u8.shape
    od = rgb_to_od(rgb_u8, I0=I0).reshape(-1, 3).T                # (3,P)
    Minv = np.linalg.pinv(W)                                      # (3,3)
    C = Minv @ od                                                 # (3,P)
    C = np.maximum(C, 0.0)                                        # nonnegative concentrations
    Cimg = C.T.reshape(H, Wd, 3)                                  # (H,W,3)
    # Return H and E OD images (higher = more stain)
    return Cimg[..., 0], Cimg[..., 1], Cimg[..., 2]

def quick_tissue_mask(tile_u8, white_thresh=240, sat_thresh=10):
    """
    Coarse tissue mask using non-white and minimal saturation.
    """
    # Non-white
    nonwhite = np.any(tile_u8 < white_thresh, axis=-1)
    # Saturation from HSV
    hsv = color.rgb2hsv(tile_u8)
    sat = (hsv[..., 1] * 255).astype(np.float32)
    tissue = nonwhite & (sat >= sat_thresh)
    # Clean
    tissue = morphology.remove_small_objects(tissue, 64)
    tissue = morphology.remove_small_holes(tissue, 64)
    return tissue