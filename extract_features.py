from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 array (H, W, 3)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return arr


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale float."""
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------- Basic steganalysis features ----------

def lsb_ratio(img_rgb: np.ndarray) -> float:
    """Ratio of LSB=1 across all RGB channels."""
    lsb = img_rgb & 1
    return float(lsb.mean())


def even_odd_imbalance(img_rgb: np.ndarray) -> float:
    vals = img_rgb.reshape(-1)
    even = np.sum((vals % 2) == 0)
    odd = vals.size - even
    return float((odd - even) / max(vals.size, 1))


def chi_square_parity(img_rgb: np.ndarray) -> float:
    vals = img_rgb.reshape(-1)
    even = np.sum((vals % 2) == 0)
    odd = vals.size - even
    expected = vals.size / 2.0
    return float(((even - expected) ** 2 + (odd - expected) ** 2) / max(expected, 1e-9))


# ---------- Residual-based features ----------

def residual_stats(gray: np.ndarray, blur_size: int = 3) -> Tuple[float, float, float]:
    blurred = uniform_filter(gray, size=blur_size, mode="reflect")
    res = gray - blurred

    res_mean = float(res.mean())
    res_std = float(res.std() + 1e-9)

    m4 = float(np.mean((res - res.mean()) ** 4))
    kurt = m4 / (res_std ** 4)
    return res_mean, res_std, float(kurt)


def neighbor_diff_stats(gray: np.ndarray) -> Tuple[float, float]:
    right = np.abs(gray[:, 1:] - gray[:, :-1])
    down = np.abs(gray[1:, :] - gray[:-1, :])
    diffs = np.concatenate([right.reshape(-1), down.reshape(-1)])
    return float(diffs.mean()), float(diffs.std())


def block_residual_variance_features(
    gray: np.ndarray, blur_size: int = 3, block: int = 8
) -> Tuple[float, float]:
    """
    Strong local feature: block-wise residual variance.
    """
    blurred = uniform_filter(gray, size=blur_size, mode="reflect")
    res = gray - blurred

    h, w = res.shape
    h2 = (h // block) * block
    w2 = (w // block) * block
    res = res[:h2, :w2]

    rb = res.reshape(h2 // block, block, w2 // block, block)
    block_var = rb.var(axis=(1, 3))

    return float(block_var.mean()), float(block_var.std())


# ---------- Feature extraction per image ----------

def extract_features_for_image(img_path: Path) -> Dict[str, float]:
    img_rgb = load_image_rgb(img_path)
    gray = to_gray(img_rgb)

    f: Dict[str, float] = {}
    f["lsb_ratio"] = lsb_ratio(img_rgb)
    f["even_odd_imbalance"] = even_odd_imbalance(img_rgb)
    f["chi_square_parity"] = chi_square_parity(img_rgb)

    res_mean, res_std, res_kurt = residual_stats(gray)
    f["res_mean"] = res_mean
    f["res_std"] = res_std
    f["res_kurtosis"] = res_kurt

    nad_mean, nad_std = neighbor_diff_stats(gray)
    f["neighbor_absdiff_mean"] = nad_mean
    f["neighbor_absdiff_std"] = nad_std

    bv_mean, bv_std = block_residual_variance_features(gray)
    f["block_res_var_mean"] = bv_mean
    f["block_res_var_std"] = bv_std

    f["gray_mean"] = float(gray.mean())
    f["gray_std"] = float(gray.std())

    return f


# ---------- Main ----------

def main() -> None:
    base = Path(__file__).resolve().parent
    dataset_dir = base / "dataset"
    labels_csv = dataset_dir / "labels.csv"
    out_csv = dataset_dir / "features.csv"

    if not labels_csv.exists():
        raise SystemExit(f"Missing: {labels_csv}")

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        items = list(reader)

    rows_out: List[Dict[str, str]] = []
    feature_names: List[str] = []

    processed = 0
    skipped = 0

    for item in items:
        rel_path = item["path"]
        label = item["label"]

        img_path = dataset_dir / rel_path
        if not img_path.exists():
            skipped += 1
            continue

        try:
            feats = extract_features_for_image(img_path)

            if not feature_names:
                feature_names = sorted(feats.keys())

            row = {"path": rel_path, "label": label}
            for k in feature_names:
                row[k] = f"{feats[k]:.10f}"

            rows_out.append(row)
            processed += 1

        except Exception as e:
            print(f"[SKIP] {rel_path}: {e}")
            skipped += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label"] + feature_names
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done. Processed={processed}, skipped={skipped}")
    print(f"Features saved to: {out_csv}")


if __name__ == "__main__":
    main()
