"""Part 3 - Edge AI triage during the scan.

A lightweight classifier (MobileNetV3-Small) flags tiles that are likely to contain
tumour *while the slide is still being scanned*. Flagged tiles keep their full z-stack
for review; everything else keeps only the fused image - saving storage without losing
the regions a pathologist cares about.

Pipeline: train on PatchCamelyon -> export ONNX -> INT8 quantise -> benchmark CPU latency.
The deployed model needs only ONNX Runtime (no PyTorch), like an embedded scanner PC.
"""

from __future__ import annotations

import time

import numpy as np

from .utils import to_float

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(tiles: list[np.ndarray] | np.ndarray, size: int = 96) -> np.ndarray:
    """uint8 RGB tiles -> normalised NCHW float32 batch."""
    import cv2

    batch = []
    for t in tiles:
        if t.shape[0] != size or t.shape[1] != size:
            t = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA)
        batch.append(((to_float(t) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
    return np.stack(batch).astype(np.float32)


# =========================================================================
# Training (PyTorch)
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

    class PCamDataset(Dataset):
        """PatchCamelyon: 96x96 H&E patches, label 1 = tumour tissue in the centre 32x32."""

        def __init__(self, x_h5: str, y_h5: str, train: bool = False, limit: int | None = None):
            import h5py

            self.x = h5py.File(x_h5, "r")["x"]
            self.y = h5py.File(y_h5, "r")["y"][:].reshape(-1)
            self.n = min(limit, len(self.y)) if limit else len(self.y)
            self.train = train

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            img = self.x[i]
            if self.train:  # flips / rotations - histology has no canonical orientation
                if np.random.rand() < 0.5:
                    img = img[:, ::-1]
                img = np.rot90(img, np.random.randint(4))
            x = torch.from_numpy(preprocess([np.ascontiguousarray(img)])[0])
            return x, torch.tensor(int(self.y[i]))

    def build_model(pretrained: bool = True) -> nn.Module:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    def train_triage(model, train_loader, val_loader, epochs: int = 3, lr: float = 1e-3,
                     device: str = "cpu") -> list[dict]:
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(train_loader))
        loss_fn = nn.CrossEntropyLoss()
        history = []
        for ep in range(epochs):
            model.train()
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if step % 200 == 0:
                    print(f"epoch {ep + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
            metrics = evaluate_triage(model, val_loader, device)
            history.append({"epoch": ep + 1, **metrics})
            print(history[-1])
        return history

    @torch.no_grad()
    def predict_probs(model, loader, device: str = "cpu"):
        model.eval().to(device)
        probs, labels = [], []
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    def evaluate_triage(model, loader, device: str = "cpu") -> dict:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        p, y = predict_probs(model, loader, device)
        return {"auc": float(roc_auc_score(y, p)),
                "acc": float(accuracy_score(y, p > 0.5)),
                "f1": float(f1_score(y, p > 0.5))}

    def export_onnx(model, path: str, size: int = 96) -> None:
        model.eval().cpu()
        torch.onnx.export(model, torch.zeros(1, 3, size, size), path,
                          input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, **_legacy_export_kw())

except ImportError:
    torch = None


# =========================================================================
# Quantisation + edge inference (ONNX Runtime only)
# =========================================================================


def quantize_int8(fp32_path: str, int8_path: str, calib_tiles: np.ndarray | None = None) -> None:
    """INT8 quantisation. Static (calibrated) if sample tiles are given, else dynamic."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(fp32_path, pre)

    if calib_tiles is None:
        quantize_dynamic(pre, int8_path, weight_type=QuantType.QInt8)
        return

    class Reader(CalibrationDataReader):
        def __init__(self, tiles):
            self.it = iter([{"image": preprocess([t])} for t in tiles])

        def get_next(self):
            return next(self.it, None)

    quantize_static(pre, int8_path, Reader(calib_tiles),
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


class OnnxTriage:
    """Tumour-probability scorer running on ONNX Runtime (CPU)."""

    def __init__(self, path: str, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, tiles) -> np.ndarray:
        logits = self.sess.run(None, {self.inp: preprocess(tiles)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True))[:, 1]


def benchmark(path: str, batch: int = 1, runs: int = 200, warmup: int = 20,
              threads: int | None = 1) -> dict:
    """CPU latency per tile (ms) and throughput (tiles/s)."""
    import os

    model = OnnxTriage(path, threads)
    tiles = (np.random.default_rng(0).random((batch, 96, 96, 3)) * 255).astype(np.uint8)
    for _ in range(warmup):
        model(tiles)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(tiles)
        times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000 / batch
    return {"model": os.path.basename(path), "size_mb": os.path.getsize(path) / 1e6,
            "ms_per_tile_p50": float(np.median(ms)), "ms_per_tile_p95": float(np.percentile(ms, 95)),
            "tiles_per_s": float(1000 / np.median(ms))}
