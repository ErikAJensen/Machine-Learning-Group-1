import os
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------- Konfigurasjon ---------
INPUT_DIR = r"C:\Users\erik\OneDrive - personlig\Skrivebord\archive (3)\raw-img"
OUTPUT_DIR = r"C:\Users\erik\OneDrive - personlig\Skrivebord\archive (3)\resized-img"  # ny mappe for ferdige bilder
 # Rotmappe der ferdige bilder lagres (samme struktur)

TARGET_SIZE = (224, 224)        
PADDING_COLOR = (0, 0, 0)       
SAVE_FORMAT = None              
QUALITY = 95                    
NUM_WORKERS = min(8, os.cpu_count() or 4)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# --------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def output_path_for(input_path: str) -> str:
    rel = os.path.relpath(input_path, INPUT_DIR)
    out = os.path.join(OUTPUT_DIR, rel)
    out_dir = os.path.dirname(out)
    ensure_dir(out_dir)
    return out

def resize_with_padding(img: Image.Image, size: tuple[int, int], fill=(0, 0, 0)) -> Image.Image:
    img = img.convert("RGB")
    img_copy = img.copy()
    img_copy.thumbnail(size, Image.LANCZOS)
    dw = size[0] - img_copy.width
    dh = size[1] - img_copy.height
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    return ImageOps.expand(img_copy, padding, fill=fill)

def process_one(file_path: str) -> str:
    try:
        with Image.open(file_path) as im:
            out_img = resize_with_padding(im, TARGET_SIZE, PADDING_COLOR)

            out_path = output_path_for(file_path)
            root, ext = os.path.splitext(out_path)

            if SAVE_FORMAT is None:
                # lagre med samme endelse der mulig; tving til .jpg om ikke støttet
                ext_lower = ext.lower()
                if ext_lower not in VALID_EXTS:
                    ext_lower = ".jpg"
                if ext_lower in (".jpg", ".jpeg"):
                    out_path = root + ".jpg"
                    out_img.save(out_path, "JPEG", quality=QUALITY, optimize=True)
                elif ext_lower == ".png":
                    out_path = root + ".png"
                    out_img.save(out_path, "PNG", optimize=True)
                else:
                    out_path = root + ".jpg"
                    out_img.save(out_path, "JPEG", quality=QUALITY, optimize=True)
            else:
                if SAVE_FORMAT.upper() == "JPEG":
                    out_path = root + ".jpg"
                    out_img.save(out_path, "JPEG", quality=QUALITY, optimize=True)
                else:
                    out_img.save(out_path, SAVE_FORMAT.upper())

        return f"OK  {file_path}"
    except Exception as e:
        return f"FEIL {file_path}: {e}"

def collect_image_files(root_dir: str):
    for cur, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                yield os.path.join(cur, f)

def main():
    ensure_dir(OUTPUT_DIR)
    files = list(collect_image_files(INPUT_DIR))
    if not files:
        print("Fant ingen bildefiler.")
        return

    print(f"Skal prosessere {len(files)} filer fra '{INPUT_DIR}' -> '{OUTPUT_DIR}' "
          f"til størrelse {TARGET_SIZE} med padding {PADDING_COLOR}.\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(process_one, p) for p in files]
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            print(f"[{i}/{len(files)}] {msg}")

    print("\nFerdig! Struktur er speilet i:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
