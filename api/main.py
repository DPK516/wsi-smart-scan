"""FastAPI service exposing the smart-scan components.

Runs with ONNX Runtime only (no PyTorch) - the same constraint as an embedded scanner PC.

    uvicorn api.main:app --reload            # then open http://localhost:8000/docs

Optional models (picked up automatically if present):
    models/blurnet.onnx        -> absolute defocus score in /quality
    models/triage_int8.onnx    -> tumour probability in /triage and /scan
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from src.fusion import fuse_focus_selection, fuse_laplacian_pyramid
from src.pipeline import ScanConfig, SmartScanner
from src.quality import FOCUS_METRICS, OnnxBlurScorer, focus_map, qc_decision, relative_blur
from src.utils import decode_rgb, tissue_mask, to_float, to_uint8

ROOT = Path(__file__).resolve().parents[1]
BLUR_MODEL = os.getenv("BLUR_MODEL", str(ROOT / "models/blurnet.onnx"))
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", str(ROOT / "models/triage_int8.onnx"))

app = FastAPI(
    title="WSI Smart Scan API",
    description="Focus QC, z-stack fusion and edge triage for whole-slide imaging.",
    version="1.0.0",
)

blur_scorer = OnnxBlurScorer(BLUR_MODEL) if Path(BLUR_MODEL).exists() else None
triage_model = None
if Path(TRIAGE_MODEL).exists():
    from src.triage import OnnxTriage
    triage_model = OnnxTriage(TRIAGE_MODEL)


async def _read(file: UploadFile) -> np.ndarray:
    try:
        return decode_rgb(await file.read())
    except ValueError:
        raise HTTPException(400, f"{file.filename} is not a readable image")


@app.get("/health")
def health():
    return {"status": "ok", "blurnet_loaded": blur_scorer is not None,
            "triage_loaded": triage_model is not None}


@app.post("/quality")
async def quality(file: UploadFile = File(...), tile: int = Query(128, ge=32),
                  threshold: float = Query(0.85, description="relative-blur threshold")):
    """Focus-quality check: global metrics, per-tile blur map, PASS / RESCAN verdict."""
    img = await _read(file)
    t0 = time.perf_counter()
    metrics = {name: fn(img) for name, fn in FOCUS_METRICS.items()}
    mask = tissue_mask(img)
    sharp = focus_map(img, tile, "laplacian_var", mask=mask)
    if np.all(np.isnan(sharp)):
        raise HTTPException(422, "no tissue tiles found - try a smaller tile size")
    qc = qc_decision(relative_blur(sharp), threshold)
    out = {
        "global_metrics": metrics,
        "tiles_with_tissue": int((~np.isnan(sharp)).sum()),
        "qc_status": qc.status,
        "bad_fraction": round(qc.bad_fraction, 4),
        "rescan_tiles": qc.bad_tiles,
        "relative_blur_grid": np.round(np.nan_to_num(qc.scores, nan=-1), 3).tolist(),
    }
    if blur_scorer is not None:
        out["blurnet_defocus_px"] = float(blur_scorer([img])[0])
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/fuse")
async def fuse(files: list[UploadFile] = File(..., description="z-stack: same field of view, 2+ focal planes"),
               method: str = Query("focus_selection", pattern="^(focus_selection|laplacian_pyramid)$")):
    """Fuse a z-stack into one all-in-focus PNG."""
    if len(files) < 2:
        raise HTTPException(400, "upload at least 2 focal planes")
    planes = [await _read(f) for f in files]
    if len({p.shape for p in planes}) != 1:
        raise HTTPException(400, "all planes must have the same size")
    stack = np.stack([to_float(p) for p in planes])
    t0 = time.perf_counter()
    if method == "laplacian_pyramid":
        fused = fuse_laplacian_pyramid(stack)
        headers = {}
    else:
        fused, idx = fuse_focus_selection(stack)
        share = np.bincount(idx.ravel(), minlength=len(planes)) / idx.size
        headers = {"X-Plane-Usage": ",".join(f"{s:.3f}" for s in share)}
    headers["X-Latency-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"

    import cv2
    ok, png = cv2.imencode(".png", cv2.cvtColor(to_uint8(fused), cv2.COLOR_RGB2BGR))
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png", headers=headers)


@app.post("/triage")
async def triage(file: UploadFile = File(...)):
    """Tumour probability for a tile (resized to 96x96)."""
    if triage_model is None:
        raise HTTPException(503, "triage model not found - run notebook 03 to create models/triage_int8.onnx")
    img = await _read(file)
    t0 = time.perf_counter()
    p = float(triage_model([img])[0])
    return {"tumour_probability": round(p, 4), "flag": p > 0.5,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/scan")
async def scan(file: UploadFile = File(..., description="sharp slide region used as ground truth"),
               tile: int = Query(64, ge=32), planes: int = Query(5, ge=2, le=15),
               seed: int = 0, failure: bool = True):
    """Simulate a full smart scan of the uploaded region and return the scan report."""
    img = await _read(file)
    cfg = ScanConfig(tile=tile, n_planes=planes, seed=seed, failure_region=failure,
                     triage_model=TRIAGE_MODEL if triage_model is not None else None)
    report = SmartScanner(cfg).run(img)
    report.pop("config")
    return report
