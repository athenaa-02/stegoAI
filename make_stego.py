from pathlib import Path
from PIL import Image


def main() -> None:
    base = Path(__file__).resolve().parent
    raw_dir = base / "dataset" / "cover_raw"
    out_dir = base / "dataset" / "cover"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise SystemExit(f"Missing folder: {raw_dir} (create it and put images inside)")

    # Keep it simple: JPG/PNG/WEBP input, output is PNG.
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    # Keep size fixed to make features comparable across images
    size = (256, 256)

    count = 0
    skipped = 0

    for p in sorted(raw_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = im.resize(size, Image.BICUBIC)

                out_path = out_dir / f"{p.stem}.png"
                im.save(out_path, format="PNG")

                count += 1
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            skipped += 1

    print(f"Done. Converted {count} images to: {out_dir} (skipped={skipped}).")


if __name__ == "__main__":
    main()

