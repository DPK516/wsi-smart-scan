"""Part 2 - Focus-stack (z-stack) fusion / extended depth of field.

Tissue on a glass slide is not perfectly flat, so a single focal plane leaves parts of
a field of view blurry. Volumetric scanners capture several focal planes (a z-stack)
and fuse the sharpest pixels into one all-in-focus image.

This module
1. **simulates** a z-stack from a sharp tile and an uneven tissue-height surface
   (defocus radius grows with distance from the focal plane), giving us a perfect
   ground truth to measure against;
2. implements two fusion methods - **per-pixel focus selection** and **Laplacian
   pyramid fusion** - plus the **single best plane** baseline;
3. evaluates them with PSNR / SSIM against the ground-truth sharp tile and measures how
   well the recovered depth map matches the true tissue surface.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .quality import disk_psf, laplacian_variance
from .utils import to_float, to_gray

# =========================================================================
# 1. Simulated acquisition
# =========================================================================


def _smooth_field(rng, h, w, sigma):
    f = filters.gaussian(rng.standard_normal((h, w)), sigma=sigma, mode="wrap")
    return f / (f.std() + 1e-8)  # unit standard deviation


def depth_surface(shape: tuple[int, int], amplitude: float = 1.0, tilt: float = 1.0,
                  roughness: float = 1.0, seed: int = 0) -> np.ndarray:
    """Random tissue-height map in *z-step units* (0 = nominal focal plane).

    = slow warp of the slide (coarse field + tilt) + local thickness changes within a
    field of view (fine field, scaled by `roughness`). The fine part is what makes a
    single focal plane insufficient and z-stack fusion necessary.
    """
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    field = amplitude * _smooth_field(rng, h, w, 0.15 * min(h, w))
    field += roughness * _smooth_field(rng, h, w, 18)
    yy, xx = np.mgrid[:h, :w]
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xx / w + np.sin(angle) * yy / h - 0.5) * tilt
    return (field + ramp).astype(np.float32)


def make_zstack(sharp: np.ndarray, depth: np.ndarray, z_planes: np.ndarray,
                blur_per_z: float = 1.5, noise: float = 0.005, seed: int = 0,
                radius_step: float = 0.5) -> np.ndarray:
    """Render a z-stack: each pixel's defocus radius = blur_per_z * |z_plane - depth|.

    Returns float32 array of shape (Z, H, W, 3) in [0, 1].
    """
    rng = np.random.default_rng(seed)
    img = to_float(sharp)
    max_r = blur_per_z * float(np.max(np.abs(z_planes[:, None, None] - depth[None])))
    radii = np.arange(0, max_r + radius_step, radius_step)
    bank = np.stack([cv2.filter2D(img, -1, disk_psf(r), borderType=cv2.BORDER_REFLECT)
                     for r in radii])  # (R, H, W, 3)

    stack = []
    for z in z_planes:
        r = blur_per_z * np.abs(z - depth) / radius_step
        lo = np.clip(np.floor(r).astype(int), 0, len(radii) - 1)
        hi = np.clip(lo + 1, 0, len(radii) - 1)
        a = (r - lo)[..., None]
        rows, cols = np.indices(depth.shape)
        layer = (1 - a) * bank[lo, rows, cols] + a * bank[hi, rows, cols]
        layer = layer + rng.normal(0, noise, layer.shape)
        stack.append(np.clip(layer, 0, 1))
    return np.stack(stack).astype(np.float32)


# =========================================================================
# 2. Fusion methods
# =========================================================================


def focus_measure(gray: np.ndarray, window: int = 9) -> np.ndarray:
    """Sum-Modified-Laplacian focus measure averaged over a local window."""
    kx = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], np.float32)
    ml = np.abs(cv2.filter2D(gray, -1, kx)) + np.abs(cv2.filter2D(gray, -1, kx.T))
    return cv2.boxFilter(ml, -1, (window, window))


def fuse_focus_selection(stack: np.ndarray, window: int = 9, smooth: int = 7):
    """Per-pixel selection of the sharpest plane.

    1. compute a local focus measure on every plane
    2. pick the plane with the highest response for each pixel (= depth index)
    3. median-filter the depth index to remove speckle, then gather the pixels

    Returns (fused RGB image, depth index map).
    """
    fm = np.stack([focus_measure(to_gray(s), window) for s in stack])
    depth_idx = np.argmax(fm, axis=0)
    if smooth > 1:
        depth_idx = ndimage.median_filter(depth_idx, size=smooth)
    rows, cols = np.indices(depth_idx.shape)
    fused = stack[depth_idx, rows, cols]
    return fused, depth_idx


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = [img]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyr = [gauss[i] - cv2.pyrUp(gauss[i + 1], dstsize=gauss[i].shape[1::-1]) for i in range(levels)]
    pyr.append(gauss[-1])
    return pyr


def fuse_laplacian_pyramid(stack: np.ndarray, levels: int = 4) -> np.ndarray:
    """Multi-scale fusion: at every pyramid level keep the coefficient with the largest
    magnitude (most detail); average the coarsest (base) level.
    Produces smoother transitions than hard per-pixel selection."""
    pyrs = [_laplacian_pyramid(s.astype(np.float32), levels) for s in stack]
    fused = []
    for lvl in range(levels):
        coeffs = np.stack([p[lvl] for p in pyrs])            # (Z, h, w, 3)
        energy = np.abs(coeffs).sum(-1)                         # pick by summed RGB magnitude
        energy = np.stack([cv2.GaussianBlur(e, (5, 5), 0) for e in energy])
        idx = np.argmax(energy, axis=0)
        rows, cols = np.indices(idx.shape)
        fused.append(coeffs[idx, rows, cols])
    fused.append(np.mean([p[-1] for p in pyrs], axis=0))
    out = fused[-1]
    for lvl in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=fused[lvl].shape[1::-1]) + fused[lvl]
    return np.clip(out, 0, 1)


def best_single_plane(stack: np.ndarray) -> tuple[np.ndarray, int]:
    """Baseline: a conventional single-plane scanner keeps the globally sharpest layer."""
    scores = [laplacian_variance(s) for s in stack]
    k = int(np.argmax(scores))
    return stack[k], k


# =========================================================================
# 3. Evaluation
# =========================================================================


def image_quality(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = to_float(pred), to_float(gt)
    return {
        "psnr": round(float(peak_signal_noise_ratio(gt, pred, data_range=1.0)), 3),
        "ssim": round(float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)), 4),
    }


def depth_error(depth_idx: np.ndarray, z_planes: np.ndarray, true_depth: np.ndarray) -> float:
    """Mean absolute error (in z-steps) between recovered and true tissue height."""
    return float(np.mean(np.abs(z_planes[depth_idx] - true_depth)))


def compare_methods(sharp: np.ndarray, n_planes: int = 7, seed: int = 0, **stack_kw) -> dict:
    """Simulate one z-stack and score every method against the ground truth."""
    z = np.arange(n_planes, dtype=np.float32) - (n_planes - 1) / 2
    depth = depth_surface(sharp.shape, seed=seed)
    depth = depth - np.median(depth)  # autofocus centres the stack on the tile
    stack = make_zstack(sharp, depth, z, seed=seed, **stack_kw)

    single, _ = best_single_plane(stack)
    sel, idx = fuse_focus_selection(stack)
    pyr = fuse_laplacian_pyramid(stack)
    return {
        "single_best_plane": image_quality(single, sharp),
        "focus_selection": {**image_quality(sel, sharp), "depth_mae": depth_error(idx, z, depth)},
        "laplacian_pyramid": image_quality(pyr, sharp),
    }
