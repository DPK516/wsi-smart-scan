"""Part 1 - Scan quality check (focus quality assessment).

Two families of methods are compared:

* **Classical focus metrics** - hand-crafted sharpness measures used in microscope
  autofocus (Laplacian variance, Tenengrad, Brenner, normalised variance).
* **BlurNet** - a tiny fully-convolutional CNN that regresses the
  defocus level of a tissue tile (~17k parameters). It is trained on *synthetic* optical defocus
  (disk point-spread function) and evaluated on *real* scanner z-stacks (FocusPath).

The output of either method feeds a per-tile pass / rescan decision and a heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import grid_shape, iter_tiles, to_float, to_gray

# =========================================================================
# 1. Classical focus metrics (higher = sharper)
# =========================================================================


def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian - responds to edges / fine texture."""
    return float(cv2.Laplacian(to_gray(img), cv2.CV_32F, ksize=3).var())


def tenengrad(img: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude."""
    g = to_gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def brenner(img: np.ndarray) -> float:
    """Brenner gradient: squared difference between pixels two apart."""
    g = to_gray(img)
    return float(np.mean((g[:, 2:] - g[:, :-2]) ** 2) + np.mean((g[2:, :] - g[:-2, :]) ** 2))


def normalized_variance(img: np.ndarray) -> float:
    """Intensity variance divided by mean - robust to illumination changes."""
    g = to_gray(img)
    return float(g.var() / (g.mean() + 1e-8))


FOCUS_METRICS = {
    "laplacian_var": laplacian_variance,
    "tenengrad": tenengrad,
    "brenner": brenner,
    "norm_variance": normalized_variance,
}


def focus_map(img: np.ndarray, tile: int, metric: str = "laplacian_var",
              mask: np.ndarray | None = None, min_tissue: float = 0.25) -> np.ndarray:
    """Per-tile focus score grid. Tiles with too little tissue are NaN."""
    fn = FOCUS_METRICS[metric]
    out = np.full(grid_shape(img.shape, tile), np.nan, dtype=np.float32)
    for y, x, t in iter_tiles(img, tile):
        if mask is not None and mask[y : y + tile, x : x + tile].mean() < min_tissue:
            continue
        out[y // tile, x // tile] = fn(t)
    return out


# =========================================================================
# 2. Synthetic optical defocus
# =========================================================================


def disk_psf(radius: float) -> np.ndarray:
    """Anti-aliased disk point-spread function - geometric model of defocus blur."""
    if radius < 0.5:
        k = np.zeros((3, 3), np.float32)
        k[1, 1] = 1.0
        return k
    r = int(np.ceil(radius)) + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(xx**2 + yy**2)
    k = np.clip(radius + 0.5 - dist, 0, 1).astype(np.float32)
    return k / k.sum()


def apply_defocus(img: np.ndarray, level: float, noise: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Blur an RGB image with a disk PSF of radius `level` pixels (+ optional sensor noise)."""
    out = cv2.filter2D(to_float(img), -1, disk_psf(level), borderType=cv2.BORDER_REFLECT)
    if noise > 0:
        rng = rng or np.random.default_rng()
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


# =========================================================================
# 3. BlurNet (PyTorch) - imported lazily so classical code works without torch
# =========================================================================
def _legacy_export_kw() -> dict:
    """Use the classic TorchScript ONNX exporter (single self-contained .onnx file)."""
    import inspect

    import torch
    return {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    class BlurNet(nn.Module):
        """Tiny fully-convolutional defocus regressor.

        Global average pooling makes it input-size agnostic, so the same weights score
        a 96 px training patch or a 512 px scanner tile.
        """

        def __init__(self, width: int = 16):
            super().__init__()

            def block(cin, cout, stride):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )

            self.features = nn.Sequential(
                block(3, width, 1),
                block(width, width, 2),
                block(width, width * 2, 1),
                block(width * 2, width * 2, 2),
            )
            self.head = nn.Conv2d(width * 2, 1, 1)

        def forward(self, x):  # x: (B, 3, H, W) in [0, 1]
            return self.head(self.features(x)).mean(dim=(2, 3)).squeeze(1)

    class SyntheticDefocusDataset(Dataset):
        """Sharp tissue patches -> (randomly defocused patch, defocus level)."""

        def __init__(self, patches: np.ndarray, max_level: float = 6.0,
                     noise: float = 0.01, seed: int = 0):
            self.patches = patches  # (N, H, W, 3) uint8
            self.max_level = max_level
            self.noise = noise
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, i):
            p = self.patches[i]
            if self.rng.random() < 0.5:
                p = p[:, ::-1]
            p = np.rot90(p, self.rng.integers(4))
            level = float(self.rng.uniform(0, self.max_level))
            img = apply_defocus(p, level, noise=self.noise, rng=self.rng)
            img = img * self.rng.uniform(0.9, 1.1)  # illumination jitter
            x = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1).copy())
            return x, torch.tensor(level, dtype=torch.float32)

    class FocusPathDataset(Dataset):
        """Real scanner blur: FocusPath patches with ground-truth z-level.

        csv format (from the FocusLiteNN repo): ``name,slide,strip,slice,position,score,z``.
        Images are 1024 px at 40x (0.25 um/px); we downsample 4x to ~1 um/px so the
        pixel scale matches PatchCamelyon (the synthetic training data).
        """

        def __init__(self, csv_path: str, root: str, downscale: int = 4, limit: int | None = None):
            import pandas as pd

            df = pd.read_csv(csv_path, header=None)
            self.names = [n.rsplit(".", 1)[0] + ".png" for n in df.iloc[:, 0]]
            self.z = np.abs(df.iloc[:, -1].to_numpy()).astype(np.float32)
            self.slide = df.iloc[:, 1].to_numpy()
            if limit:
                idx = np.random.default_rng(0).choice(len(self.names), limit, replace=False)
                self.names = [self.names[i] for i in idx]
                self.z, self.slide = self.z[idx], self.slide[idx]
            self.root, self.downscale = root, downscale

        def __len__(self):
            return len(self.names)

        def load(self, i) -> np.ndarray:
            from .utils import load_rgb

            img = load_rgb(f"{self.root}/{self.names[i]}")
            h, w = img.shape[:2]
            return cv2.resize(img, (w // self.downscale, h // self.downscale),
                              interpolation=cv2.INTER_AREA)

        def __getitem__(self, i):
            img = to_float(self.load(i))
            return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.tensor(self.z[i])

    def train_blurnet(model, train_loader, val_loader=None, epochs: int = 10,
                      lr: float = 2e-3, device: str = "cpu") -> list[dict]:
        """Standard regression loop (Smooth-L1 loss, AdamW, cosine LR)."""
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        loss_fn = nn.SmoothL1Loss()
        history = []
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(x)
            sched.step()
            row = {"epoch": ep + 1, "train_loss": tot / len(train_loader.dataset)}
            if val_loader is not None:
                pred, true = predict_loader(model, val_loader, device)
                row["val_mae"] = float(np.mean(np.abs(pred - true)))
            history.append(row)
            print(row)
        return history

    @torch.no_grad()
    def predict_loader(model, loader, device: str = "cpu"):
        model.eval().to(device)
        preds, trues = [], []
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            trues.append(y.numpy())
        return np.concatenate(preds), np.concatenate(trues)

    def export_blurnet_onnx(model, path: str, size: int = 96) -> None:
        model.eval().cpu()
        dummy = torch.zeros(1, 3, size, size)
        torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["defocus"],
                          dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"}, "defocus": {0: "batch"}},
                          opset_version=17, **_legacy_export_kw())

except ImportError:  # torch not installed - classical metrics still work
    torch = None


# =========================================================================
# 4. ONNX inference + QC decision (no torch needed - used by the API/pipeline)
# =========================================================================


class OnnxBlurScorer:
    """Run an exported BlurNet with ONNX Runtime. Returns defocus level (higher = blurrier)."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, tiles: list[np.ndarray]) -> np.ndarray:
        x = np.stack([to_float(t).transpose(2, 0, 1) for t in tiles]).astype(np.float32)
        return self.sess.run(None, {self.inp: x})[0].reshape(-1)


@dataclass
class QCResult:
    status: str                      # "PASS" or "RESCAN"
    bad_fraction: float              # share of tissue tiles that failed
    bad_tiles: list = field(default_factory=list)  # (row, col) grid indices
    scores: np.ndarray | None = None


def qc_decision(blur_grid: np.ndarray, threshold: float, max_bad_fraction: float = 0.05) -> QCResult:
    """Slide-level decision from a per-tile *blur* grid (higher = blurrier, NaN = no tissue).

    A tile fails if its blur score exceeds ``threshold``. The slide is sent for rescan if
    more than ``max_bad_fraction`` of tissue tiles fail - mirroring scanner-side QA that
    flags a scan before it reaches the pathologist.
    """
    valid = ~np.isnan(blur_grid)
    bad = valid & (blur_grid > threshold)
    frac = float(bad.sum() / max(valid.sum(), 1))
    return QCResult(
        status="RESCAN" if frac > max_bad_fraction else "PASS",
        bad_fraction=frac,
        bad_tiles=[tuple(map(int, rc)) for rc in np.argwhere(bad)],
        scores=blur_grid,
    )


def relative_blur(sharpness_grid: np.ndarray) -> np.ndarray:
    """Turn a classical *sharpness* grid into a relative *blur* score.

    Classical metrics have no absolute scale (they depend on tissue content), so we
    compare each tile to the slide median: 0 = as sharp as a typical tile, 1 = no detail.
    """
    med = np.nanmedian(sharpness_grid)
    return 1.0 - np.clip(sharpness_grid / (med + 1e-12), 0, 1)
