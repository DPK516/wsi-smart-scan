"""Download the public datasets used in this project.

    python data/download_data.py                 # PCam valid + test  (~1.6 GB) - enough for the project
    python data/download_data.py --pcam-train    # + PCam train       (~6.1 GB)
    python data/download_data.py --focuspath     # + FocusPath        (~13.1 GB) real scanner z-levels

Datasets
--------
* PatchCamelyon (PCam) - 327k 96x96 H&E lymph-node patches, tumour / normal labels.
  Veeling et al., MICCAI 2018. https://github.com/basveeling/pcam  (CC0)
  Used for: BlurNet synthetic-defocus training + MobileNet edge triage.
* FocusPath - 8,640 1024x1024 patches from 9 stained slides scanned at 14 z-levels
  (Huron TissueScope, 40x, 0.25 um/px). Hosseini et al., IEEE TIP 2019. (CC BY 4.0)
  Used for: evaluating focus-quality methods on *real* optical defocus.
  Labels come from the FocusLiteNN repo (Wang et al., MICCAI 2020).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PCAM_ZENODO = "https://zenodo.org/records/2546921/files/{name}?download=1"
PCAM_GDRIVE = {  # fallback mirrors (same IDs torchvision uses)
    "camelyonpatch_level_2_split_train_x.h5.gz": "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
    "camelyonpatch_level_2_split_train_y.h5.gz": "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
    "camelyonpatch_level_2_split_test_x.h5.gz": "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
    "camelyonpatch_level_2_split_test_y.h5.gz": "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
}
FOCUSPATH_ZIP = "https://zenodo.org/records/3926181/files/FocusPath%20Full.zip?download=1"
FOCUSPATH_LABELS = ("https://raw.githubusercontent.com/icbcbicc/FocusLiteNN/master/"
                    "data/FocusPath_full_split1.txt")


def _progress(block, size, total):
    if total > 0:
        done = min(block * size / total, 1.0)
        print(f"\r  {done:6.1%} of {total / 1e9:.2f} GB", end="", flush=True)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"✓ {dest.name} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, _progress)
    tmp.rename(dest)
    print()
    return dest


def gunzip(path: Path) -> Path:
    out = path.with_suffix("")  # strip .gz
    if not out.exists():
        print(f"  extracting {path.name}")
        with gzip.open(path, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 24)
    return out


def get_pcam(splits: list[str]) -> None:
    folder = ROOT / "pcam"
    for split in splits:
        for kind in ("x", "y"):
            name = f"camelyonpatch_level_2_split_{split}_{kind}.h5.gz"
            if (folder / name[:-3]).exists():
                print(f"✓ {name[:-3]} already present")
                continue
            try:
                gz = fetch(PCAM_ZENODO.format(name=name), folder / name)
            except Exception as e:  # Zenodo throttling -> Google Drive mirror
                print(f"  Zenodo failed ({e}); trying Google Drive via gdown")
                import gdown
                gz = folder / name
                gdown.download(id=PCAM_GDRIVE[name], output=str(gz), quiet=False)
            gunzip(gz)
            gz.unlink(missing_ok=True)


def get_focuspath() -> None:
    folder = ROOT / "focuspath"
    fetch(FOCUSPATH_LABELS, folder / "FocusPath_full_split1.txt")
    if not any(folder.rglob("*.png")):
        z = fetch(FOCUSPATH_ZIP, folder / "FocusPath_Full.zip")
        print("  unzipping FocusPath (this takes a while)")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        z.unlink()
    print(f"✓ FocusPath images under {folder}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcam-train", action="store_true", help="also download the 6 GB train split")
    ap.add_argument("--focuspath", action="store_true", help="download FocusPath (13 GB)")
    ap.add_argument("--skip-pcam", action="store_true")
    a = ap.parse_args()

    if not a.skip_pcam:
        get_pcam(["valid", "test"] + (["train"] if a.pcam_train else []))
    if a.focuspath:
        get_focuspath()
    print("Done.")
