# resize_dataset_recursive.py
# Kjører i VS Code. Krever: pip install pillow

from pathlib import Path
from PIL import Image, ImageOps, ImageFile
import sys
import time

# --- KONFIGURASJON ------------------------------------------------------------
# Pek til dine faktiske mapper (rå strenger r"...", så mellomrom/paranteser funker)
INPUT_DIR  = Path(r"C:\Users\erik1\OneDrive\Desktop\archive (3)\raw-img")
OUTPUT_DIR = Path(r"C:\Users\erik1\OneDrive\Desktop\archive (3)\resized")

# Støttede filtyper (kan utvides)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Maks lengste side (px). Bildet skaleres proporsjonalt slik at
# max(width, height) <= MAX_SIDE. Endre hvis du vil.
MAX_SIDE = 1024

# Kvalitet/lagring
JPEG_QUALITY = 90     # 1-95/100 (90 er ofte et bra kompromiss)
WEBP_QUALITY = 85     # 1-100
OVERWRITE = False     # True for å altid regenerere selv om output finnes

# Håndter delvis korrupte filer i datasett
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -----------------------------------------------------------------------------


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def newer_or_missing(src: Path, dst: Path) -> bool:
    """Returner True hvis dst mangler eller src er nyere enn dst."""
    if not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def target_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """Beregn ny størrelse slik at lengste side = max_side (eller mindre)."""
    if w <= max_side and h <= max_side:
        return w, h
    if w >= h:
        scale = max_side / float(w)
    else:
        scale = max_side / float(h)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def save_image(img: Image.Image, dst: Path) -> None:
    ext = dst.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        # Sørg for RGB (no alpha) for JPEG
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    elif ext == ".png":
        img.save(dst, format="PNG", optimize=True)
    elif ext == ".webp":
        # WEBP kan være RGB/RGBA
        img.save(dst, format="WEBP", quality=WEBP_QUALITY, method=6)
    elif ext in {".tif", ".tiff"}:
        # Komprimert TIFF (valgfritt)
        img.save(dst, format="TIFF", compression="tjpeg", quality=JPEG_QUALITY)
    else:
        # Fallback: lagre i originalt format
        img.save(dst)


def resize_one(src: Path, dst_root: Path) -> bool:
    """Resizer én fil. Returnerer True hvis prosessert, False hvis hoppet/feilet."""
    try:
        rel = src.relative_to(INPUT_DIR)
        dst = dst_root / rel
        ensure_parent(dst)

        if not OVERWRITE and not newer_or_missing(src, dst):
            # Allerede generert og oppdatert
            return False

        with Image.open(src) as im:
            # Bevar EXIF-rotasjon
            im = ImageOps.exif_transpose(im)

            w, h = im.size
            nw, nh = target_size(w, h, MAX_SIDE)

            if (nw, nh) != (w, h):
                # thumbnail endrer objektet in-place, bevarer proporsjoner
                im = im.copy()
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

            save_image(im, dst)
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
    skipped = 0

    for idx, src in enumerate(images, 1):
        ok = resize_one(src, OUTPUT_DIR)
        if ok:
            processed += 1
        else:
            skipped += 1
        if idx % 100 == 0:
            print(f"...{idx}/{len(images)} ferdig")

    elapsed = time.time() - start
    print("\n--- Oppsummering ---")
    print(f"Prosessert: {processed}")
    print(f"Hoppet over: {skipped} (allerede oppdatert eller feilet)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Tid: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
