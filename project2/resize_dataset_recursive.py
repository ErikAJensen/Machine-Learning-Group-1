# resize_dataset_recursive.py
# Krever: pip install pillow

from pathlib import Path
from PIL import Image, ImageOps, ImageFile
import sys
import time

# --- KONFIGURASJON ------------------------------------------------------------
INPUT_DIR  = Path(r"C:\Users\erik1\OneDrive\Desktop\archive (3)\raw-img")
OUTPUT_DIR = Path(r"C:\Users\erik1\OneDrive\Desktop\archive (3)\resized")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# FAST mål-oppløsning – alle bilder blir nøyaktig denne størrelsen
TARGET_SIZE = (300, 200)     # (bredde, høyde)

JPEG_QUALITY = 90
WEBP_QUALITY = 85
OVERWRITE = True             # regenerer alt

ImageFile.LOAD_TRUNCATED_IMAGES = True
# -----------------------------------------------------------------------------


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def to_fixed_size(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Skaler bildet proporsjonalt og pad til nøyaktig størrelse."""
    target_w, target_h = size
    img = ImageOps.exif_transpose(img)

    # Skaler slik at bildet passer INN i target uten å kutte
    fitted = ImageOps.contain(img, size, Image.Resampling.LANCZOS)

    # Lag hvit bakgrunn (RGB)
    background = Image.new("RGB", size, (255, 255, 255))
    x = (target_w - fitted.width) // 2
    y = (target_h - fitted.height) // 2
    background.paste(fitted, (x, y))
    return background


def save_image(img: Image.Image, dst: Path) -> None:
    ext = dst.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    elif ext == ".png":
        img.save(dst, format="PNG", optimize=True)
    elif ext == ".webp":
        img.save(dst, format="WEBP", quality=WEBP_QUALITY, method=6)
    else:
        img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def resize_one(src: Path, dst_root: Path) -> bool:
    try:
        rel = src.relative_to(INPUT_DIR)
        dst = dst_root / rel
        ensure_parent(dst)

        with Image.open(src) as im:
            fixed = to_fixed_size(im, TARGET_SIZE)
            save_image(fixed, dst)
        return True

    except Exception as e:
        print(f"[FEIL] {src} -> {e}", file=sys.stderr)
        return False


def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if is_image(p)]


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input-mappe finnes ikke: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = collect_images(INPUT_DIR)
    if not images:
        print("Fant ingen bildefiler.")
        return

    print(f"Fant {len(images)} bildefiler under {INPUT_DIR}")
    for p in images[:5]:
        print("  -", p)

    start = time.time()
    processed = 0
    for idx, src in enumerate(images, 1):
        if resize_one(src, OUTPUT_DIR):
            processed += 1
        if idx % 100 == 0:
            print(f"...{idx}/{len(images)} ferdig")

    elapsed = time.time() - start
    print(f"\nFerdig! {processed} bilder prosessert på {elapsed:.1f}s.")
    print(f"Alle bilder er nøyaktig {TARGET_SIZE[0]}x{TARGET_SIZE[1]} piksler.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
