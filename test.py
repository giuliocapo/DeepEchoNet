import os, glob, time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ============================================================
# CONFIG
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arch_type = "echo_s"  # "echo_xxs" | "echo_xs" | "echo_s"
weights_path = "Echo_runs/echo_s_64x64_run3/checkpoints/echo_s_64x64_run3_best.pth"
test_root = "/mnt/g/archive/nyu_data_64x64/data/nyu2_test"

INPUT_SIZE = (64, 64)
EPS = 1e-8
MIN_CM, MAX_CM = 50.0, 1000.0
USE_IMAGENET_NORM = True

# ============================================================
# MODELLO
# ============================================================
from architecture import DeepEchoNet

model = DeepEchoNet(device=device, arch_type=arch_type).to(device)

print(f"[INFO] Carico pesi da: {weights_path}")
sd = torch.load(weights_path, map_location=device)
model.load_state_dict(sd, strict=True)
model.eval()
torch.set_grad_enabled(False)

# ============================================================
# ⚡ MISURA INFERENZA
# ============================================================
n_warmup, n_runs, batch_size = 10, 100, 1
dummy_input = torch.randn(batch_size, 3, *INPUT_SIZE).to(device)

for _ in range(n_warmup):
    _ = model(dummy_input)
torch.cuda.synchronize()

start_time = time.time()
for _ in range(n_runs):
    _ = model(dummy_input)
torch.cuda.synchronize()
elapsed = time.time() - start_time

avg_time = elapsed / n_runs
fps = batch_size / avg_time
print(f"\n⚡ [INFERENCE SPEED] {fps:.2f} FPS ({avg_time*1000:.2f} ms/frame)\n")

# ============================================================
# PREPROCESS — coerente con il training
# ============================================================
preprocess_list = [T.Resize(INPUT_SIZE), T.ToTensor()]
if USE_IMAGENET_NORM:
    preprocess_list.append(T.Normalize(mean=[0.485,0.456,0.406],
                                       std=[0.229,0.224,0.225]))
preprocess = T.Compose(preprocess_list)

# ============================================================
# GT loader e metriche
# ============================================================
def load_gt_cm(gt_png_path: str) -> np.ndarray:
    arr = np.array(Image.open(gt_png_path).convert("I")).astype(np.float32)
    maxv = float(arr.max())
    gt_cm = arr / 10.0 if maxv > 1000.0 else (arr / 255.0) * 1000.0
    return np.clip(gt_cm, MIN_CM, MAX_CM)

def compute_metrics_cm(gt_cm, pred_cm):
    mask = np.isfinite(gt_cm) & np.isfinite(pred_cm) & (gt_cm > 0)
    if not np.any(mask):
        return [np.nan]*6
    g, p = gt_cm[mask], pred_cm[mask]
    rmse = np.sqrt(np.mean((p-g)**2))
    mae  = np.mean(np.abs(p-g))
    rel  = np.mean(np.abs(p-g)/(g+EPS))
    thr  = np.maximum(p/(g+EPS), g/(p+EPS))
    d1, d2, d3 = np.mean(thr<1.25), np.mean(thr<1.25**2), np.mean(thr<1.25**3)
    return rmse, mae, rel, d1, d2, d3

def meanv(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")

# ============================================================
# LOOP DI TEST
# ============================================================
rgb_list = sorted(glob.glob(os.path.join(test_root, "**", "*_colors.png"), recursive=True))
if not rgb_list:
    raise FileNotFoundError(f"Nessun file trovato in {test_root}")

rmseL, maeL, relL, d1L, d2L, d3L = [], [], [], [], [], []
debug_once = True

for rgb_path in tqdm(rgb_list, desc="Testing NYU Depth v2"):
    gt_path = rgb_path.replace("_colors.png", "_depth.png")
    if not os.path.exists(gt_path):
        continue

    x = preprocess(Image.open(rgb_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_raw = model(x)
        if isinstance(pred_raw, (tuple, list)):
            pred_raw = pred_raw[0]

    pred_cm = pred_raw.squeeze().cpu().numpy()
    if pred_cm.max() < 20:
        pred_cm *= 100.0  # in metri → cm
    pred_cm = np.clip(pred_cm, MIN_CM, MAX_CM)

    gt_cm_full = load_gt_cm(gt_path)
    gt_t = torch.from_numpy(gt_cm_full).unsqueeze(0).unsqueeze(0).float()
    gt_cm_low = F.interpolate(gt_t, size=pred_raw.shape[-2:], mode="bilinear", align_corners=False)
    gt_cm_low = gt_cm_low.squeeze().numpy()

    if debug_once:
        print(f"\n[DEBUG] {os.path.basename(rgb_path)}")
        print(f"   pred_cm: min={pred_cm.min():.2f}, max={pred_cm.max():.2f}, mean={pred_cm.mean():.2f}")
        print(f"   gt_cm:   min={gt_cm_low.min():.2f}, max={gt_cm_low.max():.2f}, mean={gt_cm_low.mean():.2f}")
        debug_once = False

    rmse, mae, rel, d1, d2, d3 = compute_metrics_cm(gt_cm_low, pred_cm)
    rmseL.append(rmse); maeL.append(mae); relL.append(rel)
    d1L.append(d1); d2L.append(d2); d3L.append(d3)

# ============================================================
# RISULTATI FINALI
# ============================================================
print("\n=== NYU Depth v2 — metriche (cm) ===")
print(f"RMSE={meanv(rmseL):.2f} | MAE={meanv(maeL):.2f} | AbsRel={meanv(relL):.3f} | "
      f"d1={meanv(d1L):.3f} | d2={meanv(d2L):.3f} | d3={meanv(d3L):.3f}")
