from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, convolve
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft2, fftshift


# ==================== HELPERS ====================
EPS = 1e-12

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
    return 0.299 * r + 0.587 * g + 0.114 * b

def safe_entropy(p: np.ndarray) -> float:
    p = p.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = np.clip(p, EPS, 1.0)
    return float(entropy(p))

def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    if a.std() < eps or b.std() < eps:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)

def finiteize_features(feats: Dict[str, float]) -> Dict[str, float]:
    """Replace NaN/inf with 0.0 so you never write bad CSV values."""
    out = {}
    for k, v in feats.items():
        v = float(v)
        if not np.isfinite(v):
            v = 0.0
        out[k] = v
    return out


# ==================== (1) LSB FEATURES ====================
def lsb_features(img_rgb: np.ndarray) -> Dict[str, float]:
    """LSB ratio + correlations + run stats + (2) adjacency flip-rates."""
    features: Dict[str, float] = {}

    lsb = img_rgb & 1
    features["lsb_ratio"] = float(lsb.mean())

    # LSB by channel
    for i, channel in enumerate(["R", "G", "B"]):
        channel_lsb = img_rgb[..., i] & 1
        features[f"lsb_{channel}"] = float(channel_lsb.mean())

    # Pairwise LSB correlations (SAFE)
    lsb_r = (img_rgb[..., 0] & 1)
    lsb_g = (img_rgb[..., 1] & 1)
    lsb_b = (img_rgb[..., 2] & 1)

    features["lsb_corr_RG"] = safe_corr(lsb_r, lsb_g)
    features["lsb_corr_RB"] = safe_corr(lsb_r, lsb_b)
    features["lsb_corr_GB"] = safe_corr(lsb_g, lsb_b)

    # Run statistics on combined LSB plane
    lsb_flat = lsb.ravel()
    if lsb_flat.size == 0:
        features["lsb_run_mean"] = 0.0
        features["lsb_run_std"] = 0.0
    else:
        runs = []
        current_run = 1
        for i in range(1, len(lsb_flat)):
            if lsb_flat[i] == lsb_flat[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        runs = np.array(runs, dtype=np.float32)
        features["lsb_run_mean"] = float(runs.mean())
        features["lsb_run_std"] = float(runs.std())

    # ==================== (2) LSB adjacency flip-rates (STRONG) ====================
    # Combined LSB plane
    lsb2d = (img_rgb & 1).astype(np.uint8)
    # Reduce (H,W,3) -> (H,W) by XOR combining channels (simple but effective)
    lsb_comb = lsb2d[..., 0] ^ lsb2d[..., 1] ^ lsb2d[..., 2]

    # Horizontal / vertical transition rates
    if lsb_comb.shape[1] > 1:
        features["lsb_flip_h"] = float((lsb_comb[:, 1:] != lsb_comb[:, :-1]).mean())
    else:
        features["lsb_flip_h"] = 0.0

    if lsb_comb.shape[0] > 1:
        features["lsb_flip_v"] = float((lsb_comb[1:, :] != lsb_comb[:-1, :]).mean())
    else:
        features["lsb_flip_v"] = 0.0

    # Per-channel flip rates (even better)
    for idx, channel in enumerate(["R", "G", "B"]):
        plane = (img_rgb[..., idx] & 1).astype(np.uint8)
        if plane.shape[1] > 1:
            features[f"lsb_flip_h_{channel}"] = float((plane[:, 1:] != plane[:, :-1]).mean())
        else:
            features[f"lsb_flip_h_{channel}"] = 0.0
        if plane.shape[0] > 1:
            features[f"lsb_flip_v_{channel}"] = float((plane[1:, :] != plane[:-1, :]).mean())
        else:
            features[f"lsb_flip_v_{channel}"] = 0.0

    return features


# ==================== (3) CLASSIC CHI-SQUARE FEATURES ====================
def chi_square_features(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Classic chi-square on histogram pairs (2k, 2k+1).
    This is a well-known strong feature for LSB replacement stego.
    """
    features: Dict[str, float] = {}

    for i, channel in enumerate(["R", "G", "B"]):
        vals = img_rgb[..., i].ravel()

        # Histogram 0..255
        hist = np.bincount(vals, minlength=256).astype(np.float64)

        # Pair chi-square
        chi = 0.0
        for k in range(0, 256, 2):
            o1 = hist[k]
            o2 = hist[k + 1]
            e = (o1 + o2) / 2.0
            if e > 0:
                chi += ((o1 - e) ** 2 + (o2 - e) ** 2) / e

        features[f"chi_sq_pairs_{channel}"] = float(chi)

        # Even/odd balance chi (you had this; keep it)
        even = np.sum((vals % 2) == 0)
        odd = vals.size - even
        expected = vals.size / 2.0
        chi_evenodd = ((even - expected) ** 2 + (odd - expected) ** 2) / max(expected, EPS)
        features[f"chi_sq_evenodd_{channel}"] = float(chi_evenodd)
        features[f"p_even_{channel}"] = float(even / max(vals.size, 1))

        # Optional: entropy of 2-byte pairs (cheap additional feature)
        if vals.size >= 2:
            pairs = vals[: (vals.size // 2) * 2].reshape(-1, 2)
            _, counts = np.unique(pairs, axis=0, return_counts=True)
            features[f"pair_entropy_{channel}"] = safe_entropy(counts.astype(np.float64))
        else:
            features[f"pair_entropy_{channel}"] = 0.0

    return features


# ==================== SPATIAL FEATURES ====================
def spatial_features(gray: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    g = gray.astype(np.float32)

    features["gray_mean"] = float(g.mean())
    features["gray_std"] = float(g.std())
    flat = g.ravel()
    features["gray_skew"] = float(skew(flat)) if flat.size > 3 else 0.0
    features["gray_kurtosis"] = float(kurtosis(flat)) if flat.size > 3 else 0.0

    hist, _ = np.histogram(g, bins=256, range=(0, 255))
    features["hist_entropy"] = safe_entropy(hist.astype(np.float64))
    s = hist.sum()
    if s > 0:
        hist_norm = hist / s
        features["hist_energy"] = float(np.sum(hist_norm ** 2))
    else:
        features["hist_energy"] = 0.0

    h, w = g.shape
    center = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]) if h > 1 and w > 1 else g.ravel()
    features["contrast_center_mean"] = float(center.mean()) if center.size else 0.0
    features["contrast_border_mean"] = float(border.mean()) if border.size else 0.0

    return features


def gradient_features(gray: np.ndarray) -> Dict[str, float]:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    g = gray.astype(np.float32)
    grad_x = convolve(g, sobel_x, mode="reflect")
    grad_y = convolve(g, sobel_y, mode="reflect")

    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    ang = np.arctan2(grad_y, grad_x)

    return {
        "grad_mag_mean": float(mag.mean()),
        "grad_mag_std": float(mag.std()),
        "grad_mag_max": float(mag.max()) if mag.size else 0.0,
        "grad_angle_mean": float(ang.mean()),
        "grad_angle_std": float(ang.std()),
    }


def residual_features(gray: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    g = gray.astype(np.float32)

    for blur in [3, 5, 7]:
        blurred = uniform_filter(g, size=blur, mode="reflect")
        residual = g - blurred

        features[f"res_mean_{blur}"] = float(residual.mean())
        features[f"res_std_{blur}"] = float(residual.std())

        centered = residual - residual.mean()
        flat = centered.ravel()
        features[f"res_skew_{blur}"] = float(skew(flat)) if flat.size > 3 else 0.0
        features[f"res_kurt_{blur}"] = float(kurtosis(flat)) if flat.size > 3 else 0.0
        features[f"res_energy_{blur}"] = float(np.mean(centered ** 2))

    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap_res = convolve(g, laplacian, mode="reflect")
    features["lap_res_mean"] = float(lap_res.mean())
    features["lap_res_std"] = float(lap_res.std())

    return features


# ==================== (4) RESIDUAL COOCCURRENCE (SRM-LITE) ====================
def residual_cooccurrence_features(gray: np.ndarray, T: int = 2) -> Dict[str, float]:
    """
    SRM-lite:
      - compute simple high-pass residuals
      - clamp to [-T..T], shift to [0..2T]
      - build co-occurrence counts for neighboring residuals
    This is usually a big accuracy booster for LSB stego.
    """
    g = gray.astype(np.float32)

    # Simple high-pass filters
    f_h = np.array([[1, -1]], dtype=np.float32)        # horizontal diff
    f_v = np.array([[1], [-1]], dtype=np.float32)      # vertical diff
    f_l = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)      # laplacian-ish

    r_h = convolve(g, f_h, mode="reflect")
    r_v = convolve(g, f_v, mode="reflect")
    r_l = convolve(g, f_l, mode="reflect")

    def cooc_2d(res: np.ndarray) -> Dict[str, float]:
        # clamp + quantize
        q = np.clip(np.rint(res), -T, T).astype(np.int32) + T  # 0..2T
        L = 2 * T + 1

        feats: Dict[str, float] = {}

        # horizontal neighbor co-occurrence
        if q.shape[1] > 1:
            a = q[:, :-1].ravel()
            b = q[:, 1:].ravel()
            idx = a * L + b
            counts = np.bincount(idx, minlength=L * L).astype(np.float64)
            feats["cooc_h_entropy"] = safe_entropy(counts)
            feats["cooc_h_energy"] = float(np.sum((counts / max(counts.sum(), 1.0)) ** 2))
        else:
            feats["cooc_h_entropy"] = 0.0
            feats["cooc_h_energy"] = 0.0

        # vertical neighbor co-occurrence
        if q.shape[0] > 1:
            a = q[:-1, :].ravel()
            b = q[1:, :].ravel()
            idx = a * L + b
            counts = np.bincount(idx, minlength=L * L).astype(np.float64)
            feats["cooc_v_entropy"] = safe_entropy(counts)
            feats["cooc_v_energy"] = float(np.sum((counts / max(counts.sum(), 1.0)) ** 2))
        else:
            feats["cooc_v_entropy"] = 0.0
            feats["cooc_v_energy"] = 0.0

        return feats

    out: Dict[str, float] = {}
    for name, res in [("H", r_h), ("V", r_v), ("L", r_l)]:
        feats = cooc_2d(res)
        for k, v in feats.items():
            out[f"{name}_{k}"] = v

    return out

# New helper – more SRM-like rich high-pass filters (small set)
def get_rich_kernels() -> List[np.ndarray]:
    """Small but effective set of SRM-inspired kernels"""
    kernels = []

    # 1st order
    kernels.append(np.array([[ 0, 0, 0],
                             [ 0, 1,-1],
                             [ 0, 0, 0]], dtype=np.float32))   # right
    kernels.append(np.array([[ 0, 0, 0],
                             [-1, 1, 0],
                             [ 0, 0, 0]], dtype=np.float32))   # left

    # 2nd order / edge
    kernels.append(np.array([[ 0, 1, 0],
                             [ 1,-4, 1],
                             [ 0, 1, 0]], dtype=np.float32))   # laplace

    kernels.append(np.array([[-1, 2,-1],
                             [ 2,-4, 2],
                             [-1, 2,-1]], dtype=np.float32))   # stronger laplace

    # KV-like (edge sensitive)
    kernels.append(np.array([[ 1, -2, 1],
                             [-2, 4,-2],
                             [ 1, -2, 1]], dtype=np.float32))

    return kernels


def srm_lite_rich_cooc(gray: np.ndarray, T: int = 3, symmetry: bool = True) -> Dict[str, float]:
    """
    More filters → more co-occurrence features.
    T=3 is common compromise (range [-3…3] → 7 levels)
    """
    g = gray.astype(np.float32)
    features = {}

    kernels = get_rich_kernels()

    for i, kern in enumerate(kernels, 1):
        res = convolve(g, kern, mode='reflect')

        q = np.clip(np.rint(res), -T, T).astype(np.int32) + T   # 0 … 2T
        L = 2 * T + 1

        # horizontal cooc
        if q.shape[1] > 1:
            a = q[:, :-1].ravel()
            b = q[:,  1:].ravel()
            idx = a * L + b
            cnt = np.bincount(idx, minlength=L*L).astype(np.float64)
            if symmetry:
                cnt += np.bincount(b * L + a, minlength=L*L).astype(np.float64)
            p = cnt / max(cnt.sum(), 1e-9)
            features[f"srm_c{i}_h_ent"]  = safe_entropy(cnt)
            features[f"srm_c{i}_h_ene"]  = float(np.sum(p**2))
            features[f"srm_c{i}_h_var"]  = float(np.sum((np.arange(L*L) % L - np.arange(L*L)//L)**2 * p))

        # vertical cooc (similar)
        if q.shape[0] > 1:
            a = q[:-1, :].ravel()
            b = q[ 1:, :].ravel()
            idx = a * L + b
            cnt = np.bincount(idx, minlength=L*L).astype(np.float64)
            if symmetry:
                cnt += np.bincount(b * L + a, minlength=L*L).astype(np.float64)
            p = cnt / max(cnt.sum(), 1e-9)
            features[f"srm_c{i}_v_ent"]  = safe_entropy(cnt)
            features[f"srm_c{i}_v_ene"]  = float(np.sum(p**2))
            features[f"srm_c{i}_v_var"]  = float(np.sum((np.arange(L*L) % L - np.arange(L*L)//L)**2 * p))

    return features

# ==================== FREQUENCY FEATURES ====================
def frequency_features(gray: np.ndarray) -> Dict[str, float]:
    g = gray.astype(np.float32)
    fft = fft2(g)
    fft_shifted = fftshift(fft)
    mag = np.abs(fft_shifted).astype(np.float64)

    h, w = mag.shape
    if h > 0 and w > 0:
        mag[h // 2, w // 2] = 0  # remove DC

    total = mag.sum()
    if total <= 0:
        ent = 0.0
    else:
        ent = safe_entropy(mag.ravel())

    # radial distribution
    y, x = np.indices(mag.shape)
    cy, cx = h // 2, w // 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    radial_profile = np.bincount(radius.ravel(), weights=mag.ravel())
    radial_counts = np.bincount(radius.ravel())
    radial_avg = radial_profile / np.maximum(radial_counts, 1)

    # avoid empty slices
    q1 = max(h // 4, 1)
    q2 = max(h // 2, 1)

    return {
        "freq_mag_mean": float(mag.mean()),
        "freq_mag_std": float(mag.std()),
        "freq_mag_entropy": float(ent),
        "freq_radial_low": float(radial_avg[:q1].mean()) if radial_avg.size else 0.0,
        "freq_radial_mid": float(radial_avg[q1:q2].mean()) if radial_avg.size else 0.0,
        "freq_radial_high": float(radial_avg[q2:].mean()) if radial_avg.size else 0.0,
    }


# ==================== COLOR FEATURES ====================
def color_features(img_rgb: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for i, channel in enumerate(["R", "G", "B"]):
        c = img_rgb[..., i].astype(np.float32).ravel()
        features[f"{channel}_mean"] = float(c.mean())
        features[f"{channel}_std"] = float(c.std())
        features[f"{channel}_skew"] = float(skew(c)) if c.size > 3 else 0.0

    r = img_rgb[..., 0].astype(np.float32).ravel()
    g = img_rgb[..., 1].astype(np.float32).ravel()
    b = img_rgb[..., 2].astype(np.float32).ravel()

    features["corr_RG"] = safe_corr(r, g)
    features["corr_RB"] = safe_corr(r, b)
    features["corr_GB"] = safe_corr(g, b)

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    features["YCbCr_Y_mean"] = float(Y.mean())
    features["YCbCr_Cb_mean"] = float(Cb.mean())
    features["YCbCr_Cr_mean"] = float(Cr.mean())

    return features


# ==================== SIMPLE GLCM (VECTORIZED) ====================
def cooccurrence_features(gray: np.ndarray, levels: int = 32) -> Dict[str, float]:
    """
    Faster, vectorized co-occurrence for multiple directions.
    """
    g = gray.astype(np.float32)
    q = (g * (levels - 1) / 255.0).astype(np.int32)

    feats: Dict[str, float] = {}

    def glcm_dir(dy: int, dx: int) -> np.ndarray:
        h, w = q.shape
        y1 = slice(max(0, dy), h + min(0, dy))
        x1 = slice(max(0, dx), w + min(0, dx))
        y2 = slice(max(0, -dy), h + min(0, -dy))
        x2 = slice(max(0, -dx), w + min(0, -dx))
        a = q[y1, x1].ravel()
        b = q[y2, x2].ravel()
        idx = a * levels + b
        counts = np.bincount(idx, minlength=levels * levels).astype(np.float64)
        com = counts.reshape(levels, levels)
        s = com.sum()
        if s > 0:
            com /= s
        return com

    # directions: right, down, diag, anti-diag
    dirs = {"H": (0, 1), "V": (1, 0), "D": (1, 1), "A": (1, -1)}

    for name, (dy, dx) in dirs.items():
        com = glcm_dir(dy, dx)
        if com.sum() <= 0:
            feats[f"com_energy_{name}"] = 0.0
            feats[f"com_contrast_{name}"] = 0.0
            feats[f"com_homogeneity_{name}"] = 0.0
            continue

        feats[f"com_energy_{name}"] = float(np.sum(com ** 2))
        diff = np.abs(np.arange(levels)[:, None] - np.arange(levels)[None, :])
        feats[f"com_contrast_{name}"] = float(np.sum((diff ** 2) * com))
        feats[f"com_homogeneity_{name}"] = float(np.sum(com / (1.0 + diff)))

    # average across directions
    feats["com_energy"] = float(np.mean([feats[f"com_energy_{k}"] for k in dirs]))
    feats["com_contrast"] = float(np.mean([feats[f"com_contrast_{k}"] for k in dirs]))
    feats["com_homogeneity"] = float(np.mean([feats[f"com_homogeneity_{k}"] for k in dirs]))

    return feats


# ==================== MAIN EXTRACTION FUNCTION ====================
def extract_features_for_image(img_path: Path) -> Dict[str, float]:
    img_rgb = load_image_rgb(img_path)
    gray = to_gray(img_rgb)

    features: Dict[str, float] = {}

    groups = [
        ("LSB", lsb_features(img_rgb)),                         # (1) + (2)
        ("ChiSq", chi_square_features(img_rgb)),                # (3)
        ("Spatial", spatial_features(gray)),
        ("Gradient", gradient_features(gray)),
        ("Residual", residual_features(gray)),
        ("ResCooc", residual_cooccurrence_features(gray, T=2)), # (4)
        ("Frequency", frequency_features(gray)),
        ("Color", color_features(img_rgb)),
        ("CoOccur", cooccurrence_features(gray, levels=32)),
        ("SRM_rich", srm_lite_rich_cooc(gray, T=3, symmetry=True))
    ]

    for group_name, group_features in groups:
        for k, v in group_features.items():
            features[f"{group_name}_{k}"] = float(v)

    # Combined ratios (guarded)
    std = max(features.get("Spatial_gray_std", 0.0), EPS)
    features["complexity"] = float(features.get("Gradient_grad_mag_mean", 0.0) / std)
    features["noise_level"] = float(features.get("Residual_res_std_3", 0.0) / std)

    return finiteize_features(features)


# ==================== MAIN ====================
def main() -> None:
    base = Path(__file__).resolve().parent
    dataset_dir = base / "dataset"
    labels_csv = dataset_dir / "labels.csv"
    out_csv = dataset_dir / "features.csv"

    if not labels_csv.exists():
        raise SystemExit(f"Missing: {labels_csv}")

    rows_out: List[Dict[str, str]] = []
    feature_names: List[str] = []

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("labels.csv must have headers: path,label")
        items = list(reader)

    total = len(items)
    processed = 0
    skipped = 0

    print("Extracting steganalysis features (upgraded)...")

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
                print(f"Total features extracted: {len(feature_names)}")

            out_row: Dict[str, str] = {"path": rel_path, "label": label}
            for k in feature_names:
                out_row[k] = f"{feats[k]:.10f}"
            rows_out.append(out_row)

            processed += 1
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total} (processed={processed}, skipped={skipped})")

        except Exception as e:
            print(f"[SKIP] {rel_path}: {e}")
            skipped += 1

    fieldnames = ["path", "label"] + feature_names
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nDone. Wrote {len(rows_out)} rows with {len(feature_names)} features to: {out_csv}")
    print(f"Processed={processed}, skipped={skipped}")

    categories = defaultdict(int)
    for fname in feature_names:
        cat = fname.split("_")[0]
        categories[cat] += 1

    print("\nFeature breakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:10s}: {count:3d} features")


if __name__ == "__main__":
    main()
