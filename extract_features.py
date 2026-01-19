from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 array of shape (H, W, 3)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return arr


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32."""
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    # Standard luminance approximation
    return 0.299 * r + 0.587 * g + 0.114 * b


def lsb_ratio(img_rgb: np.ndarray) -> float:
    """Ratio of 1s in the LSBs across all RGB channels."""
    lsb = img_rgb & 1
    return float(lsb.mean())


def even_odd_imbalance(img_rgb: np.ndarray) -> float:
    """
    Measures imbalance between even and odd pixel values.
    For perfectly balanced parity, this is near 0.
    """
    vals = img_rgb.reshape(-1)
    even = np.sum((vals % 2) == 0)
    odd = vals.size - even
    return float((odd - even) / max(vals.size, 1))


def chi_square_parity(img_rgb: np.ndarray) -> float:
    """
    Simple chi-square statistic comparing counts of even vs odd.
    Stego tends to push parity toward balance depending on embedding.
    """
    vals = img_rgb.reshape(-1)
    even = np.sum((vals % 2) == 0)
    odd = vals.size - even
    expected = vals.size / 2.0
    # avoid divide-by-zero
    chi = ((even - expected) ** 2 + (odd - expected) ** 2) / max(expected, 1e-9)
    return float(chi)


def residual_stats(gray: np.ndarray, blur_size: int = 3) -> Tuple[float, float, float]:
    """
    Compute stats of residual = gray - blur(gray).
    Returns (res_mean, res_std, res_kurtosis_approx).
    """
    # uniform_filter is fast and stable
    blurred = uniform_filter(gray, size=blur_size, mode="reflect")
    res = gray - blurred

    res_mean = float(res.mean())
    res_std = float(res.std() + 1e-9)

    # Simple kurtosis (not unbiased; good enough for features)
    m4 = float(np.mean((res - res.mean()) ** 4))
    kurt = m4 / (res_std ** 4)
    return res_mean, res_std, float(kurt)


def neighbor_diff_stats(gray: np.ndarray) -> Tuple[float, float]:
    """
    Neighbor differences catch correlation changes.
    Returns (mean_abs_diff, std_abs_diff) using right and down neighbors.
    """
    right = np.abs(gray[:, 1:] - gray[:, :-1])
    down = np.abs(gray[1:, :] - gray[:-1, :])
    diffs = np.concatenate([right.reshape(-1), down.reshape(-1)])
    return float(diffs.mean()), float(diffs.std())


def extract_features_for_image(img_path: Path) -> Dict[str, float]:
    img_rgb = load_image_rgb(img_path)
    gray = to_gray(img_rgb)

    f: Dict[str, float] = {}
    f["lsb_ratio"] = lsb_ratio(img_rgb)
    f["even_odd_imbalance"] = even_odd_imbalance(img_rgb)
    f["chi_square_parity"] = chi_square_parity(img_rgb)

    res_mean, res_std, res_kurt = residual_stats(gray, blur_size=3)
    f["res_mean"] = res_mean
    f["res_std"] = res_std
    f["res_kurtosis"] = res_kurt

    mad, sad = neighbor_diff_stats(gray)
    f["neighbor_absdiff_mean"] = mad
    f["neighbor_absdiff_std"] = sad

    # Basic global stats (cheap, often helpful)
    f["gray_mean"] = float(gray.mean())
    f["gray_std"] = float(gray.std())

    return f


def main() -> None:
    base = Path(__file__).resolve().parent
    dataset_dir = base / "dataset"
    labels_csv = dataset_dir / "labels.csv"
    out_csv = dataset_dir / "features.csv"

    if not labels_csv.exists():
        raise SystemExit(f"Missing: {labels_csv}")

    rows_out: List[Dict[str, str]] = []
    feature_names: List[str] = []

    # Read labels
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("labels.csv must have headers: path,label")

        items = list(reader)

    total = len(items)
    processed = 0
    skipped = 0

    for i, item in enumerate(items, start=1):
        rel_path = item["path"].strip()
        label = item["label"].strip()

        img_path = dataset_dir / rel_path
        if not img_path.exists():
            print(f"[SKIP] Missing file: {img_path}")
            skipped += 1
            continue

        try:
            feats = extract_features_for_image(img_path)

            if not feature_names:
                feature_names = sorted(feats.keys())

            out_row: Dict[str, str] = {"path": rel_path, "label": label}
            for k in feature_names:
                out_row[k] = f"{feats[k]:.10f}"
            rows_out.append(out_row)

            processed += 1
            if i % 100 == 0 or i == total:
                print(f"Progress: {i}/{total} (processed={processed}, skipped={skipped})")

        except Exception as e:
            print(f"[SKIP] {rel_path}: {e}")
            skipped += 1

    # Write features.csv
    fieldnames = ["path", "label"] + feature_names
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nDone. Wrote {len(rows_out)} rows to: {out_csv}")
    print(f"Processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
