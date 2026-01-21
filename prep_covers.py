


from __future__ import annotations

import csv
import secrets
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image


# ---------- bytes -> bits helpers ----------

def bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def int_to_32bits(n: int) -> List[int]:
    if n < 0 or n >= 2**32:
        raise ValueError("Length must fit in 32 bits.")
    out: List[int] = []
    for i in range(31, -1, -1):
        out.append((n >> i) & 1)
    return out


# ---------- LSB embedding ----------

def embed_lsb_rgb(image: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed payload bits into the LSB of each RGB channel.
    Stores payload length (in bits) in the first 32 bits.
    Capacity = width * height * 3 bits.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Pillow new API (avoids deprecation warning)
    pixels = list(image.get_flattened_data())  # [(R,G,B), ...]
    capacity_bits = len(pixels) * 3

    payload_bits = bytes_to_bits(payload)
    header_bits = int_to_32bits(len(payload_bits))
    bits = header_bits + payload_bits

    if len(bits) > capacity_bits:
        raise ValueError(
            f"Payload too large: need {len(bits)} bits, capacity {capacity_bits} bits. "
            f"Use larger images or smaller payload_bytes."
        )

    bit_idx = 0
    new_pixels = []

    for (r, g, b) in pixels:
        if bit_idx < len(bits):
            r = (r & 0xFE) | bits[bit_idx]
            bit_idx += 1
        if bit_idx < len(bits):
            g = (g & 0xFE) | bits[bit_idx]
            bit_idx += 1
        if bit_idx < len(bits):
            b = (b & 0xFE) | bits[bit_idx]
            bit_idx += 1
        new_pixels.append((r, g, b))

    stego = Image.new("RGB", image.size)
    stego.putdata(new_pixels)
    return stego


# ---------- Dataset generation ----------

def iter_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> None:
    base = Path(__file__).resolve().parent
    dataset_dir = base / "dataset"
    cover_dir = dataset_dir / "cover"
    stego_dir = dataset_dir / "stego"
    stego_dir.mkdir(parents=True, exist_ok=True)

    if not cover_dir.exists():
        raise SystemExit(f"Missing folder: {cover_dir}")

    # CHANGE THIS for experiments:
    # 4096, 8192, 16384 are good values for 256x256 images (max ~24576 bytes).
    payload_bytes = 16384

    labels_csv = dataset_dir / "labels.csv"
    rows: List[Tuple[str, int]] = []

    count = 0
    skipped = 0

    for idx, img_path in enumerate(iter_images(cover_dir), start=1):
        print(f"[{idx}] Processing: {img_path.name}")
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img.load()

            payload = secrets.token_bytes(payload_bytes)
            stego = embed_lsb_rgb(img, payload)

            out_name = f"{img_path.stem}_stego.png"
            out_path = stego_dir / out_name

            # Save as PNG to preserve LSBs
            stego.save(out_path, format="PNG")

            rows.append((str((Path("cover") / img_path.name).as_posix()), 0))
            rows.append((str((Path("stego") / out_name).as_posix()), 1))

            count += 1

        except Exception as e:
            print(f"[SKIP] {img_path.name}: {e}")
            skipped += 1

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

    print(f"\nDone. Processed {count} cover images (skipped={skipped}).")
    print(f"Stego images saved to: {stego_dir}")
    print(f"Labels CSV saved to: {labels_csv}")


if __name__ == "__main__":
    main()
