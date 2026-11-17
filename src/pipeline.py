"""End-to-end simulated 'smart scan' of a slide.

    overview -> tissue detection -> (per tissue tile, streaming)
        acquire z-stack -> fuse -> focus QC -> edge triage -> keep / drop z-stack
    -> slide report (QC verdict, rescan tiles, storage saved, latency) + figures

The 'scanner' is simulated: we start from a sharp slide image, generate an uneven tissue
surface (plus an optional focus-failure region such as a tissue fold), and render the
z-stack each tile would produce. Because the sharp image is known, every stage can be
measured against ground truth.

Usage
-----
    python -m src.pipeline --demo --out results/demo
    python -m src.pipeline --image my_slide.png --tile 256 --triage models/triage_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .fusion import (best_single_plane, depth_surface, fuse_focus_selection,
                     fuse_laplacian_pyramid, image_quality, make_zstack)
from .quality import OnnxBlurScorer, laplacian_variance, qc_decision, relative_blur
from .utils import grid_shape, overlay_heatmap, tile_grid, tissue_mask, to_uint8


@dataclass
class ScanConfig:
    tile: int = 64
    n_planes: int = 5                 # focal planes per z-stack
    blur_per_z: float = 1.5           # defocus radius (px) per z-step away from focus
    surface_amplitude: float = 1.5    # slow slide warp (z-steps, std)
    roughness: float = 1.0            # local thickness variation inside a tile (z-steps, std)
    failure_region: bool = True       # inject an autofocus failure on a cluster of tiles
    af_error: float = 5.0             # z-offset of the failed autofocus
    min_tissue: float = 0.3           # skip tiles with less tissue than this
    fusion: str = "focus_selection"   # or "laplacian_pyramid"
    qc_threshold: float = 0.85        # relative blur (classical; tuned on seeds 0-3) or defocus px (BlurNet)
    max_bad_fraction: float = 0.05
    blur_model: str | None = None     # optional BlurNet ONNX
    triage_model: str | None = None   # optional MobileNet ONNX
    triage_threshold: float = 0.5
    seed: int = 0


class SmartScanner:
    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        self.blur = OnnxBlurScorer(cfg.blur_model) if cfg.blur_model else None
        if cfg.triage_model:
            from .triage import OnnxTriage
            self.triage = OnnxTriage(cfg.triage_model)
        else:
            self.triage = None

    # ------------------------------------------------------------- simulation
    def tissue_surface(self, shape) -> np.ndarray:
        c = self.cfg
        return depth_surface(shape, amplitude=c.surface_amplitude, roughness=c.roughness, seed=c.seed)

    def autofocus_errors(self, tissue_g: np.ndarray) -> np.ndarray:
        """Per-tile autofocus error (z-steps). Small jitter everywhere; optionally a
        cluster of tiles where autofocus locks onto the wrong plane (e.g. debris or an
        air bubble on the coverslip) - the classic cause of out-of-focus scans."""
        c = self.cfg
        rng = np.random.default_rng(c.seed + 1)
        err = rng.normal(0, 0.2, tissue_g.shape)
        cand = np.argwhere(tissue_g >= c.min_tissue)
        if c.failure_region and len(cand):
            r, k = cand[rng.integers(len(cand))]
            err[max(r - 1, 0):r + 1, max(k - 1, 0):k + 2] += c.af_error
        return err

    # ------------------------------------------------------------------- scan
    def run(self, slide: np.ndarray) -> dict:
        c, t = self.cfg, self.cfg.tile
        H, W = (slide.shape[0] // t) * t, (slide.shape[1] // t) * t
        slide = slide[:H, :W]
        z = np.arange(c.n_planes, dtype=np.float32) - (c.n_planes - 1) / 2
        depth = self.tissue_surface((H, W))

        # 1. low-res overview -> tissue mask (what the scanner does before scanning)
        small = cv2.resize(slide, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(tissue_mask(small, min_size=50).astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

        g = grid_shape((H, W), t)
        tissue_g = np.array([[mask[y:y + t, x:x + t].mean() for x in range(0, W, t)]
                             for y in range(0, H, t)])
        af_err = self.autofocus_errors(tissue_g)
        sharp_g, blur_g, triage_g = (np.full(g, np.nan) for _ in range(3))
        z_true_g, z_est_g = np.full(g, np.nan), np.full(g, np.nan)
        fused_slide = np.full((H, W, 3), 245 / 255, np.float32)
        single_slide = fused_slide.copy()
        keep_stack = np.zeros(g, bool)
        timings = []

        # 2. stream tiles in scan order
        for y, x in tile_grid((H, W), t):
            r, k = y // t, x // t
            if tissue_g[r, k] < c.min_tissue:
                continue
            # autofocus centres the z-stack on the tile's tissue height (+ error)
            planes = z + float(np.median(depth[y:y + t, x:x + t])) + af_err[r, k]
            stack = make_zstack(slide[y:y + t, x:x + t], depth[y:y + t, x:x + t], planes,
                                blur_per_z=c.blur_per_z, seed=c.seed + r * 1000 + k)
            t_acq = time.perf_counter()
            if c.fusion == "laplacian_pyramid":
                fused = fuse_laplacian_pyramid(stack)
                idx = None
            else:
                fused, idx = fuse_focus_selection(stack)
            single, _ = best_single_plane(stack)
            fused_u8 = to_uint8(fused)

            sharp_g[r, k] = laplacian_variance(fused_u8)
            if self.blur is not None:
                blur_g[r, k] = float(self.blur([fused_u8])[0])
            if self.triage is not None:
                triage_g[r, k] = float(self.triage([fused_u8])[0])
            timings.append((time.perf_counter() - t_acq) * 1000)  # processing only

            if idx is not None:
                z_est_g[r, k] = float(np.mean(planes[idx]))
            z_true_g[r, k] = float(depth[y:y + t, x:x + t].mean())
            fused_slide[y:y + t, x:x + t] = fused
            single_slide[y:y + t, x:x + t] = single

        # 3. focus QC on the fused result
        if self.blur is None:
            blur_g = relative_blur(sharp_g)
        qc = qc_decision(blur_g, c.qc_threshold, c.max_bad_fraction)

        # 4. keep the z-stack only where it matters (triage-flagged or failed QC)
        bad = ~np.isnan(blur_g) & (blur_g > c.qc_threshold)
        keep_stack |= bad
        if self.triage is not None:
            keep_stack |= np.nan_to_num(triage_g) > c.triage_threshold

        # ground truth: tiles whose autofocus failed (we injected them, so we know)
        tissue_tiles = ~np.isnan(sharp_g)
        failed = tissue_tiles & (np.abs(af_err) > 2)
        flagged = bad
        qc_eval = {
            "true_af_failures": int(failed.sum()),
            "detected": int((failed & flagged).sum()),
            "false_alarms": int((~failed & flagged & tissue_tiles).sum()),
        }
        ok = np.kron(tissue_tiles & ~flagged, np.ones((t, t))).astype(bool)[..., None]

        # 5. storage + quality summary
        tile_bytes = t * t * 3
        n_total = g[0] * g[1]
        n_tissue = int((~np.isnan(sharp_g)).sum())
        full = n_total * c.n_planes * tile_bytes          # naive: every tile, every plane
        smart = (n_tissue + int(keep_stack.sum()) * (c.n_planes - 1)) * tile_bytes
        tissue_px = np.kron(tissue_tiles, np.ones((t, t))).astype(bool)
        gt = slide.astype(np.float32) / 255
        report = {
            "config": asdict(c),
            "tiles_total": int(n_total),
            "tiles_tissue": n_tissue,
            "tiles_skipped_background": int(n_total - n_tissue),
            "tiles_zstack_kept": int(keep_stack.sum()),
            "qc_status": qc.status,
            "qc_bad_fraction": round(qc.bad_fraction, 4),
            "rescan_tiles": qc.bad_tiles,
            "storage_full_mb": round(full / 1e6, 2),
            "storage_smart_mb": round(smart / 1e6, 2),
            "storage_saved_pct": round(100 * (1 - smart / full), 1),
            "ms_per_tile_processing": round(float(np.mean(timings)), 2) if timings else None,
            "qc_vs_ground_truth": qc_eval,
            "quality_all_tissue_tiles": {
                "single_plane": image_quality(single_slide * tissue_px[..., None], gt * tissue_px[..., None]),
                "fused": image_quality(fused_slide * tissue_px[..., None], gt * tissue_px[..., None]),
            },
            "quality_qc_passed_tiles": {
                "single_plane": image_quality(single_slide * ok, gt * ok),
                "fused": image_quality(fused_slide * ok, gt * ok),
            },
        }
        if not np.all(np.isnan(z_est_g)):
            report["depth_mae_tile_zsteps"] = round(float(np.nanmean(np.abs(z_est_g - z_true_g))), 3)

        self.last = dict(slide=slide, mask=mask, depth=depth, fused=fused_slide,
                         single=single_slide, blur=blur_g, triage=triage_g,
                         keep=keep_stack, z_true=z_true_g, z_est=z_est_g)
        return report

    # ---------------------------------------------------------------- figures
    def save_figures(self, out: str | Path) -> None:
        import matplotlib.pyplot as plt

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        d, t = self.last, self.cfg.tile

        fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
        ax[0].imshow(d["slide"]); ax[0].contour(d["mask"], [0.5], colors="cyan", linewidths=1.2)
        ax[0].set_title("1 · Overview + tissue detection")
        ax[1].imshow(to_uint8(d["single"])); ax[1].set_title("2a · Best single focal plane / tile")
        ax[2].imshow(to_uint8(d["fused"])); ax[2].set_title("2b · Z-stack fusion (all-in-focus)")
        ax[3].imshow(overlay_heatmap(to_uint8(d["fused"]), d["blur"], t, cmap="RdYlGn_r", vmin=0, vmax=1))
        for r, k in np.argwhere(~np.isnan(d["blur"]) & (d["blur"] > self.cfg.qc_threshold)):
            ax[3].add_patch(plt.Rectangle((k * t, r * t), t, t, fill=False, ec="red", lw=2))
        ax[3].set_title("3 · Focus QC heatmap (red box = rescan)")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(out / "pipeline_overview.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # zoomed before / after on the blurriest region of the single-plane result
        sp = np.array([[laplacian_variance(to_uint8(d["single"][y:y + t, x:x + t]))
                        if not np.isnan(d["blur"][y // t, x // t]) else np.inf
                        for x in range(0, d["single"].shape[1], t)]
                       for y in range(0, d["single"].shape[0], t)])
        fr = np.abs(sp - np.nanmedian(sp[np.isfinite(sp)]))
        fr[~np.isfinite(sp)] = -1
        fr[np.nan_to_num(d["blur"], nan=1) > self.cfg.qc_threshold] = -1
        r, k = np.unravel_index(np.argmax(fr), fr.shape)
        y, x = r * t, k * t
        crops = [d["slide"][y:y + t, x:x + t], to_uint8(d["single"][y:y + t, x:x + t]),
                 to_uint8(d["fused"][y:y + t, x:x + t])]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
        for a, im, title in zip(ax, crops, ["Ground truth (sharp)", "Best single plane", "Fused z-stack"]):
            a.imshow(im); a.set_title(title); a.axis("off")
        plt.tight_layout()
        plt.savefig(out / "fusion_zoom.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
        vmin, vmax = np.nanmin(d["z_true"]), np.nanmax(d["z_true"])
        ax[0].imshow(d["z_true"], cmap="viridis", vmin=vmin, vmax=vmax); ax[0].set_title("True tissue height (z)")
        im = ax[1].imshow(d["z_est"], cmap="viridis", vmin=vmin, vmax=vmax); ax[1].set_title("Recovered from z-stack")
        for a in ax:
            a.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.025)
        plt.savefig(out / "depth_map.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        if not np.all(np.isnan(d["triage"])):
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(overlay_heatmap(to_uint8(d["fused"]), d["triage"], t, cmap="magma", vmin=0, vmax=1))
            plt.title("4 · Edge triage: tumour probability"); plt.axis("off")
            plt.savefig(out / "triage_heatmap.png", dpi=110, bbox_inches="tight")
            plt.close(fig)


def demo_slide(size: int = 768, seed: int = 0) -> np.ndarray:
    """Synthetic slide built from scikit-image's bundled IHC tissue sample."""
    from skimage import data

    from .utils import make_synthetic_slide
    return make_synthetic_slide(data.immunohistochemistry(), size=size, seed=seed)


def benchmark_seeds(seeds, out: str | Path, **cfg_kw) -> dict:
    """Run the simulated scan on several random slides/surfaces and aggregate metrics.
    Seeds 0-3 were used to pick the QC threshold - report on other seeds."""
    rows = []
    for sd in seeds:
        rep = SmartScanner(ScanConfig(seed=sd, **cfg_kw)).run(demo_slide(seed=sd))
        q, g = rep["quality_qc_passed_tiles"], rep["qc_vs_ground_truth"]
        rows.append({"seed": sd, "tissue_tiles": rep["tiles_tissue"],
                     "psnr_single": q["single_plane"]["psnr"], "psnr_fused": q["fused"]["psnr"],
                     "ssim_single": q["single_plane"]["ssim"], "ssim_fused": q["fused"]["ssim"],
                     "af_failures": g["true_af_failures"], "detected": g["detected"],
                     "false_alarms": g["false_alarms"], "storage_saved_pct": rep["storage_saved_pct"],
                     "depth_mae": rep.get("depth_mae_tile_zsteps"),
                     "ms_per_tile": rep["ms_per_tile_processing"]})
    keys = [k for k in rows[0] if k != "seed" and rows[0][k] is not None]
    summary = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    summary["detection_rate"] = round(sum(r["detected"] for r in rows) / max(sum(r["af_failures"] for r in rows), 1), 4)
    summary["false_alarm_rate"] = round(sum(r["false_alarms"] for r in rows) /
                                        sum(r["tissue_tiles"] - r["af_failures"] for r in rows), 4)
    result = {"seeds": list(seeds), "summary": summary, "runs": rows}
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "pipeline_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="Simulated intelligent WSI scan")
    p.add_argument("--image", help="sharp slide / region image (PNG, JPG, TIF)")
    p.add_argument("--demo", action="store_true", help="use built-in synthetic slide")
    p.add_argument("--out", default="results/demo")
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--planes", type=int, default=5)
    p.add_argument("--fusion", default="focus_selection", choices=["focus_selection", "laplacian_pyramid"])
    p.add_argument("--blur-model")
    p.add_argument("--triage")
    p.add_argument("--qc-threshold", type=float)
    p.add_argument("--no-failure", action="store_true")
    p.add_argument("--blur-per-z", type=float, default=1.5,
                   help="defocus radius (px) per z-step; scale with pixel size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", help="seed range, e.g. 10-19: run many simulated slides")
    a = p.parse_args()

    if a.benchmark:
        lo, hi = map(int, a.benchmark.split("-"))
        res = benchmark_seeds(range(lo, hi + 1), a.out, blur_per_z=a.blur_per_z, n_planes=a.planes,
                              fusion=a.fusion, tile=a.tile)
        print(json.dumps(res["summary"], indent=2))
        return

    from .utils import load_rgb
    slide = demo_slide(seed=a.seed) if a.demo or not a.image else load_rgb(a.image)
    cfg = ScanConfig(tile=a.tile, n_planes=a.planes, fusion=a.fusion, blur_model=a.blur_model,
                     triage_model=a.triage, failure_region=not a.no_failure, seed=a.seed,
                     blur_per_z=a.blur_per_z)
    if a.qc_threshold is not None:
        cfg.qc_threshold = a.qc_threshold
    elif a.blur_model:
        cfg.qc_threshold = 2.0  # BlurNet outputs defocus radius in px

    scanner = SmartScanner(cfg)
    report = scanner.run(slide)
    scanner.save_figures(a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "scan_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("config", "rescan_tiles")}, indent=2))


if __name__ == "__main__":
    main()
