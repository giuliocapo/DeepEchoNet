import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def inspect_images(root, pattern="*.*", limit=None):
    img_paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not img_paths:
        print(f"[WARN] Nessuna immagine trovata in {root}")
        return []

    print(f"[INFO] Trovate {len(img_paths)} immagini in {root}")
    results = []
    for i, path in enumerate(tqdm(img_paths, desc=f"Scansione {root}")):
        if limit and i >= limit:
            break
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Impossibile leggere {path}")
                continue
            h, w = img.shape[:2]
            c = 1 if img.ndim == 2 else img.shape[2]
            dtype = str(img.dtype)
            vmin, vmax = float(np.min(img)), float(np.max(img))
            vmean = float(np.mean(img))
            ext = Path(path).suffix.lower()
            results.append({
                "path": path,
                "ext": ext,
                "shape": f"{h}x{w}x{c}",
                "dtype": dtype,
                "min": vmin,
                "max": vmax,
                "mean": vmean,
            })
        except Exception as e:
            print(f"[ERR] {path}: {e}")
    return results


def summarize(results, name):
    if not results:
        return
    df = pd.DataFrame(results)
    print(f"\n=== {name} ===")
    print(f"Totale immagini: {len(df)}")
    print("Formati unici:", df['ext'].unique())
    print("Dtypes unici:", df['dtype'].unique())
    print("Shape uniche:", df['shape'].value_counts().head(5).to_dict())
    print(f"Valore min medio: {df['min'].mean():.2f} | max medio: {df['max'].mean():.2f} | mean medio: {df['mean'].mean():.2f}")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Analizza struttura e formato delle immagini in un dataset RGB+depth.")
    ap.add_argument("--rgb_root", required=True, help="Cartella contenente le immagini RGB (es. *_colors.png)")
    ap.add_argument("--depth_root", required=True, help="Cartella contenente le depth (es. *_depth.png)")
    ap.add_argument("--out_csv", default="dataset_summary.csv", help="File CSV di output")
    ap.add_argument("--limit", type=int, default=None, help="Numero massimo di file da analizzare (debug)")
    args = ap.parse_args()

    rgb_results = inspect_images(args.rgb_root, pattern="*_colors.*", limit=args.limit)
    depth_results = inspect_images(args.depth_root, pattern="*_depth.*", limit=args.limit)

    df_rgb = summarize(rgb_results, "RGB")
    df_depth = summarize(depth_results, "DEPTH")

    if df_rgb is not None and df_depth is not None:
        df_rgb["type"] = "rgb"
        df_depth["type"] = "depth"
        df_all = pd.concat([df_rgb, df_depth])
        df_all.to_csv(args.out_csv, index=False)
        print(f"\n✅ Report salvato in {args.out_csv}")


if __name__ == "__main__":
    main()
