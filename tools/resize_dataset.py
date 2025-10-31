import argparse
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

# Formati accettati
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ============================================================
# FUNZIONI UTILI
# ============================================================
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def all_images(root: Path):
    """Ritorna lista di tutte le immagini ricorsivamente."""
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            yield p

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def resize_one(src_path: Path, dst_path: Path, out_w: int, out_h: int, jpg_quality: int = 95):
    """Ridimensiona un file singolo mantenendo il tipo (8/16 bit)."""
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Impossibile leggere: {src_path}")
        return

    # Interpolazione: AREA per RGB, NEAREST per depth
    if img.ndim == 3 and img.shape[2] == 3:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_NEAREST

    h, w = img.shape[:2]
    if (h, w) != (out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=interp)

    ensure_dir(dst_path)

    ext = src_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
    else:
        cv2.imwrite(str(dst_path), img)

def resize_split(src_root: Path, dst_root: Path, out_w: int, out_h: int, jpg_quality: int = 95):
    """Ridimensiona un intero split (train o test)."""
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()

    imgs = list(all_images(src_root))
    if not imgs:
        print(f"[WARN] Nessuna immagine trovata in {src_root}")
        return

    print(f"\n[INFO] ➤ Ridimensiono {len(imgs)} immagini da {src_root} → {dst_root} ({out_w}×{out_h})")

    for src in tqdm(imgs, desc=f"Resizing {src_root.name}", ncols=80):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        resize_one(src, dst, out_w, out_h, jpg_quality)

    print(f"[OK] Completato: {dst_root}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ridimensiona dataset (train/test) mantenendo formati e canali.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--src_root", type=str,
                   help="Root che contiene sottocartelle 'train' e 'test'.")
    g.add_argument("--src_train_test", nargs=2, metavar=("SRC_TRAIN", "SRC_TEST"),
                   help="Percorsi separati per train e test.")
    parser.add_argument("--dst_root", type=str, default=None,
                        help="Cartella di output (opzionale).")
    parser.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=[64, 64],
                        help="Altezza e larghezza di output (default: 64 64).")
    parser.add_argument("--jpg_quality", type=int, default=95,
                        help="Qualità JPEG (default 95).")

    args = parser.parse_args()
    out_h, out_w = args.size

    # Caso 1: src_root → train/test
    if args.src_root:
        src_root = Path(args.src_root).resolve()
        train_dir = src_root / "train"
        test_dir  = src_root / "test"
        if not train_dir.exists() or not test_dir.exists():
            print("[ERRORE] Con --src_root mi aspetto sottocartelle 'train' e 'test'.", file=sys.stderr)
            sys.exit(1)

        dst_root = Path(args.dst_root) if args.dst_root else src_root.with_name(f"{src_root.name}_{out_h}x{out_w}")
        dst_train = dst_root / "train"
        dst_test  = dst_root / "test"

        print(f"[INFO] Output principale: {dst_root}")
        resize_split(train_dir, dst_train, out_w, out_h, args.jpg_quality)
        resize_split(test_dir,  dst_test,  out_w, out_h, args.jpg_quality)

    # Caso 2: train e test separati
    else:
        src_train = Path(args.src_train_test[0]).resolve()
        src_test  = Path(args.src_train_test[1]).resolve()

        if args.dst_root:
            dst_root = Path(args.dst_root).resolve()
            dst_train = dst_root / "train"
            dst_test  = dst_root / "test"
        else:
            dst_train = src_train.with_name(f"{src_train.name}_{out_h}x{out_w}")
            dst_test  = src_test.with_name(f"{src_test.name}_{out_h}x{out_w}")

        print(f"[INFO] Output: {dst_train}  |  {dst_test}")
        resize_split(src_train, dst_train, out_w, out_h, args.jpg_quality)
        resize_split(src_test,  dst_test,  out_w, out_h, args.jpg_quality)

    print("\n✅ Tutto completato con successo!")

if __name__ == "__main__":
    main()
