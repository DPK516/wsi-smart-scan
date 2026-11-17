"""Shared helpers: image I/O, colour spaces, tissue detection, tiling, plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy import ndimage
from skimage import color, filters, morphology


# --------------------------------------------------------------------------- I/O
def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image from disk as uint8 RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_rgb(buffer: bytes) -> np.ndarray:
    """Decode an encoded image (PNG/JPG/TIF bytes) to uint8 RGB."""
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(img: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR))


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0, 1]."""
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


# -------------------------------------------------------------------- colour
def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB -> float32 grayscale in [0, 1]."""
    img = to_float(img)
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    return color.rgb2hsv(to_float(img))


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    return color.rgb2lab(to_float(img))


def tissue_mask(img: np.ndarray, min_size: int = 500, closing: int = 5) -> np.ndarray:
    """Detect tissue vs. glass background.

    Stained tissue is coloured (saturation) and/or darker (value) while empty glass is
    near-white. We combine both cues from HSV into one score, threshold it with Otsu,
    then clean the mask with morphology.
    """
    hsv = rgb_to_hsv(img)
    score = filters.gaussian(hsv[..., 1] + (1 - hsv[..., 2]), sigma=2)
    glass = np.percentile(score, 5)            # brightest, least-coloured pixels
    # Otsu alone can split dark stain vs. pale stain; cap it just above the glass level
    thr = min(filters.threshold_otsu(score), glass + 0.08)
    mask = score > thr
    mask = ndimage.binary_closing(mask, structure=morphology.disk(closing))
    mask = _drop_small(mask, min_size)
    mask = ~_drop_small(~mask, min_size)  # fill small holes
    return mask


def _drop_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


# -------------------------------------------------------------------- tiling
def tile_grid(shape: tuple[int, int], tile: int) -> list[tuple[int, int]]:
    """Top-left corners of non-overlapping tiles that fully fit in the image."""
    h, w = shape[:2]
    return [(y, x) for y in range(0, h - tile + 1, tile) for x in range(0, w - tile + 1, tile)]


def iter_tiles(img: np.ndarray, tile: int) -> Iterator[tuple[int, int, np.ndarray]]:
    for y, x in tile_grid(img.shape, tile):
        yield y, x, img[y : y + tile, x : x + tile]


def grid_shape(shape: tuple[int, int], tile: int) -> tuple[int, int]:
    return shape[0] // tile, shape[1] // tile


# ------------------------------------------------------------ synthetic slide
def make_synthetic_slide(tissue: np.ndarray, size: int = 768, seed: int = 0) -> np.ndarray:
    """Place a tissue image (native resolution, centred) inside an irregular blob on a
    white 'glass' background.

    Used for demos when no real whole-slide image is available (e.g. with scikit-image's
    bundled IHC sample). Keeping native resolution preserves the fine detail that
    focus metrics depend on.
    """
    rng = np.random.default_rng(seed)
    th, tw = tissue.shape[:2]
    tex = cv2.copyMakeBorder(tissue, (size - th) // 2, size - th - (size - th) // 2,
                             (size - tw) // 2, size - tw - (size - tw) // 2,
                             cv2.BORDER_REFLECT)[:size, :size]

    noise = filters.gaussian(rng.standard_normal((size, size)), sigma=size / 12)
    yy, xx = np.mgrid[:size, :size] / size - 0.5
    blob = -(xx**2 + yy**2) * 6 + noise * 12
    mask = blob > np.quantile(blob, 0.55)
    mask = filters.gaussian(mask.astype(float), sigma=3)[..., None]

    glass = np.full_like(tex, 245, dtype=np.uint8)
    slide = mask * tex.astype(float) + (1 - mask) * glass.astype(float)
    return slide.astype(np.uint8)


# ------------------------------------------------------------------- plotting
def grid_to_image(values: np.ndarray, tile: int) -> np.ndarray:
    """Upsample a per-tile grid to pixel resolution (nearest neighbour)."""
    return np.kron(values, np.ones((tile, tile)))


def overlay_heatmap(img: np.ndarray, grid: np.ndarray, tile: int,
                    cmap: str = "RdYlGn", alpha: float = 0.45,
                    vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Blend a per-tile score grid over the image. NaN cells stay transparent."""
    import matplotlib

    heat = grid_to_image(grid, tile)
    valid = ~np.isnan(heat)
    lo = np.nanmin(grid) if vmin is None else vmin
    hi = np.nanmax(grid) if vmax is None else vmax
    norm = np.clip((np.nan_to_num(heat, nan=lo) - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = matplotlib.colormaps[cmap](norm)[..., :3]

    base = to_float(img)[: heat.shape[0], : heat.shape[1]]
    out = base.copy()
    out[valid] = (1 - alpha) * base[valid] + alpha * rgb[valid]
    return to_uint8(out)
