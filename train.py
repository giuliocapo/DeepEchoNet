# ============================================================
# train.py — METER training dinamico (scratch o fine-tuning)
# ============================================================
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from architecture import DeepEchoNet
from augmentation import augmentation2D
from loss import balanced_loss_function

# =========================
# CONFIG principale
# =========================
EXP_NAME            = "echo_s_64x64_run4"   # come vuoi salvare i file
ARCH_TYPE           = "echo_s"                    # 'echo_s' 'xxs' | 'xs' 

# input del MODELLO (non del dataset!): i METER “loro” usano 192×256
INPUT_HW            = (64, 64)             # (H, W)

# dataset CSV (relativi alla root sotto)
TRAIN_CSV           = "/mnt/g/archive/nyu_data_64x64/nyu2_train.csv"
TEST_CSV            = "/mnt/g/archive/nyu_data_64x64/nyu2_test.csv"
ROOT_BASE           = "/mnt/g/archive/nyu_data_64x64"

# run mode
START_FROM_CKPT     = False                  # True=fine-tuning, False=da zero
CKPT_PATH           = "models/build_model_best_nyu_s"  # usato se START_FROM_CKPT=True

# normalizzazione input
USE_IMAGENET_NORM   = True                  # metti True se i pesi/esperimenti lo richiedono

# augmentation (se stai usando immagini piccole tipo 64×64, disattiva il random crop nel tuo 'globals')
# altrimenti lascia gestire a augmentation2D

# optimizer & scheduler
SEED                = 42
BATCH_SIZE_PER_GPU  = 256
GLOBAL_BATCH_TARGET = 256
EPOCHS              = 200                    # per FT bastano poche epoche
LR_INIT             = 1e-3                   # scratch #il loro ufficiale è 1e-3
LR_FT               = 1e-4                   # fine-tuning
STEP_SIZE           = 20                     # scratch
STEP_SIZE_FT        = 5                      # fine-tuning
GAMMA               = 0.1
WEIGHT_DECAY        = 0.01
BETAS               = (0.9, 0.999)
USE_DYNAMIC_LR = True                        # True → ReduceLROnPlateau, False → StepLR classico


# depth range & numerics
MIN_CM, MAX_CM      = 50.0, 1000.0
EPS                 = 1e-8

# output dirs & previews
OUT_DIR             = f"Echo_runs/{EXP_NAME}"
CKPT_DIR            = os.path.join(OUT_DIR, "checkpoints")
PREVIEW_DIR         = os.path.join(OUT_DIR, "previews")
PREVIEW_INTERVAL    = 5                      # salva preview ogni N epoche
PREVIEW_SAMPLES     = 4                      # quante triplette RGB/GT/PRED salvare

# =========================
# UTILS
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_meters(arr_cm: torch.Tensor) -> torch.Tensor:
    return arr_cm / 100.0

def compute_metrics_m(pred_m: torch.Tensor, gt_m: torch.Tensor):
    """RMSE/REL/d1 in METRI."""
    with torch.no_grad():
        mask = torch.isfinite(gt_m) & torch.isfinite(pred_m) & (gt_m > 0)
        if not torch.any(mask):
            return float('nan'), float('nan'), float('nan')
        p, g = pred_m[mask], gt_m[mask]
        rmse = torch.sqrt(torch.mean((g - p) ** 2)).item()
        rel  = torch.mean(torch.abs(g - p) / (g + EPS)).item()
        thr  = torch.max(g / (p + EPS), p / (g + EPS))
        d1   = (thr < 1.25).float().mean().item()
        return rmse, rel, d1

def denorm_rgb_if_needed(rgb_bchw: torch.Tensor) -> torch.Tensor:
    """Se è stata applicata la normalizzazione ImageNet, inverti per le preview."""
    if not USE_IMAGENET_NORM:  # [0,1]
        return torch.clamp(rgb_bchw, 0, 1)
    mean = torch.tensor([0.485,0.456,0.406], device=rgb_bchw.device)[None, :, None, None]
    std  = torch.tensor([0.229,0.224,0.225], device=rgb_bchw.device)[None, :, None, None]
    x = rgb_bchw * std + mean
    return torch.clamp(x, 0, 1)

def colormap_depth_cm(d_cm: np.ndarray, vmin=MIN_CM, vmax=MAX_CM):
    """mappa cm → RGB con colormap tipo 'plasma' usando matplotlib (evita cv2 in BGR)."""
    import matplotlib
    import matplotlib.cm as cm
    d = np.clip(d_cm, vmin, vmax)
    d = (d - vmin) / max(vmax - vmin, 1e-6)
    colored = cm.get_cmap('plasma')(d)[:, :, :3]  # [0,1] RGB
    return (colored * 255).astype(np.uint8)

def save_val_previews(model, val_loader, device, epoch, outdir=PREVIEW_DIR, n_samples=PREVIEW_SAMPLES):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for rgb, depth_cm in val_loader:
            rgb = rgb.to(device)          # [B,3,H,W], H,W = INPUT_HW
            depth_cm = depth_cm.to(device)  # [B,1,H,W] (già INPUT_HW dal Dataset)

            pred_cm = model(rgb)
            if isinstance(pred_cm, (tuple, list)):
                pred_cm = pred_cm[0]      # [B,1,hp,wp]

            # ---- riallinea TUTTO alla size dell’RGB ----
            H, W = rgb.shape[-2], rgb.shape[-1]
            if pred_cm.shape[-2:] != (H, W):
                pred_cm_up = F.interpolate(pred_cm, size=(H, W), mode="bilinear", align_corners=False)
            else:
                pred_cm_up = pred_cm

            if depth_cm.shape[-2:] != (H, W):
                depth_cm_up = F.interpolate(depth_cm, size=(H, W), mode="nearest")
            else:
                depth_cm_up = depth_cm

            # clamp per visualizzazione
            pred_cm_up = torch.clamp(pred_cm_up, MIN_CM, MAX_CM)
            depth_cm_up = torch.clamp(depth_cm_up, MIN_CM, MAX_CM)

            # prepara per salvataggio
            rgb_vis = denorm_rgb_if_needed(rgb).cpu().numpy()              # [B,3,H,W] in [0,1]
            gt_cm   = depth_cm_up.squeeze(1).cpu().numpy()                 # [B,H,W] cm
            pd_cm   = pred_cm_up.squeeze(1).cpu().numpy()                  # [B,H,W] cm

            B = rgb_vis.shape[0]
            for b in range(B):
                if saved >= n_samples: return

                rgb_hwc  = (np.transpose(rgb_vis[b], (1,2,0)) * 255).astype(np.uint8)
                gt_rgb   = colormap_depth_cm(gt_cm[b])     # [H,W,3] uint8 RGB
                pred_rgb = colormap_depth_cm(pd_cm[b])     # [H,W,3] uint8 RGB

                # concat orizzontale: RGB | GT | PRED
                trio = np.concatenate([rgb_hwc, gt_rgb, pred_rgb], axis=1)

                out_path = os.path.join(outdir, f"ep{epoch:03d}_sample{saved:02d}.png")
                # cv2 vuole BGR
                cv2.imwrite(out_path, cv2.cvtColor(trio, cv2.COLOR_RGB2BGR))
                saved += 1

            if saved >= n_samples:
                return

# =========================
# DATASET
# =========================
class NYUDatasetCSV(Dataset):
    """Dataset NYU da CSV (path relativi). Con resize a INPUT_HW e (opzionale) normalize ImageNet."""
    def __init__(self, csv_file: str, root_base: str, is_train: bool = True, print_info_aug=False):
        self.is_train = is_train
        self.root_base = root_base
        self.print_info_aug = print_info_aug
        self.H, self.W = INPUT_HW

        df = pd.read_csv(csv_file, header=None)
        assert df.shape[1] >= 2, f"CSV {csv_file} deve avere 2 colonne (rgb, depth)"
        self.rgb_rel   = df.iloc[:, 0].tolist()
        self.depth_rel = df.iloc[:, 1].tolist()

        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_rgb(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB non trovato: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    def _load_depth_cm(self, path):
        dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if dep is None:
            raise FileNotFoundError(f"Depth non trovata: {path}")
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        dep = dep.astype(np.float32)
        maxv = float(dep.max())
        if dep.dtype == np.uint16 or maxv > 1000.0:
            dep_cm = dep / 10.0
        else:
            dep_cm = (dep / 255.0) * 1000.0
        dep_cm = np.clip(dep_cm, MIN_CM, MAX_CM)[..., None]
        return dep_cm

    def __getitem__(self, idx):
        rgb_path   = os.path.join(self.root_base, self.rgb_rel[idx])
        depth_path = os.path.join(self.root_base, self.depth_rel[idx])

        rgb   = self._load_rgb(rgb_path)
        depth = self._load_depth_cm(depth_path)

        if self.is_train:
            rgb, depth = augmentation2D(rgb, depth, self.print_info_aug)

        if (rgb.shape[0], rgb.shape[1]) != (self.H, self.W):
            rgb   = cv2.resize(rgb,   (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            print(f"ATTENZIONE RISOLUZIONE IMMAGINI: [DEBUG] RGB dtype: {rgb.dtype} | Depth dtype: {depth.dtype}")

        if USE_IMAGENET_NORM:
            rgb = (rgb - self.imagenet_mean) / self.imagenet_std

        if depth.ndim == 2:
            depth = depth[..., None]
        depth = np.clip(depth, MIN_CM, MAX_CM)

        rgb_t   = torch.from_numpy(rgb.transpose(2, 0, 1)).float()      # [3,H,W]
        depth_t = torch.from_numpy(depth.transpose(2, 0, 1)).float()    # [1,H,W]
        return rgb_t, depth_t

    def __len__(self):
        return len(self.rgb_rel)

# =========================
# DATALOADERS
# =========================
def get_loaders(train_csv, test_csv, root_base, batch_per_gpu=BATCH_SIZE_PER_GPU, num_workers=4):
    train_set = NYUDatasetCSV(train_csv, root_base, True)
    val_set   = NYUDatasetCSV(test_csv,  root_base, False)

    train_loader = DataLoader(train_set, batch_size=batch_per_gpu, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=batch_per_gpu, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    print(f"[DEBUG] Train set: {len(train_set)} | Test set: {len(val_set)}")
    return train_loader, val_loader

# =========================
# TRAIN LOOP
# =========================
def _load_ckpt_if_any(model, device):
    if not START_FROM_CKPT:
        print("[INFO] Training from scratch.")
        return None
    sd = torch.load(CKPT_PATH, map_location=device)
    if isinstance(sd, dict):
        for k in ["ema","model_ema","model_ema_state_dict","state_dict","model","net"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    print(f"[INFO] Fine-tuning da checkpoint: {CKPT_PATH}")
    return True

def train_meter():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(TRAIN_CSV, TEST_CSV, ROOT_BASE)

    model = DeepEchoNet(device=device, arch_type=ARCH_TYPE).to(device)

    # =========== architecture summary =========== 
    from tools.architecture_summary import summarize_model
    arch_md = summarize_model(model, input_size=(1,3,64,64), arch_name=f"METER_{ARCH_TYPE}",
                          save_path=os.path.join(OUT_DIR, f"architecture_{ARCH_TYPE}.md"))  

    # ckpt & LR
    loss_curve, rel_curve, lr_curve = [], [], []

    finetuned = _load_ckpt_if_any(model, device)
    lr = LR_FT if finetuned else LR_INIT
    step_size = STEP_SIZE_FT if finetuned else STEP_SIZE

    criterion = balanced_loss_function(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    if USE_DYNAMIC_LR:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=0.0001,
        cooldown=2,
        min_lr=1e-6,
        )
        print("[LR] ✅ Using dynamic scheduler: ReduceLROnPlateau")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=GAMMA)
        print(f"[LR] ✅ Using static scheduler: StepLR(step_size={step_size}, gamma={GAMMA})")


    accum_steps = max(1, math.ceil(GLOBAL_BATCH_TARGET / BATCH_SIZE_PER_GPU))
    print(f"[INFO] Batch={BATCH_SIZE_PER_GPU}, AccumSteps={accum_steps} → eff. batch ≈{BATCH_SIZE_PER_GPU*accum_steps}")
    print(f"[INFO] LR iniziale: {lr} | StepLR step_size={step_size}, gamma={GAMMA}")
    print(f"[INFO] Previews ogni {PREVIEW_INTERVAL} epoche in {PREVIEW_DIR}")

    best_rel = float('inf')

    no_improve_epochs = 0
    early_stop_patience = 10   # fermati se non migliora per 10 epoche
    collapse_counter = 0


    # -------- Sanity check iniziale --------
    model.eval()
    with torch.no_grad():
        rgb, depth_cm = next(iter(val_loader))
        rgb, depth_cm = rgb.to(device), depth_cm.to(device)
        pred_cm = model(rgb)
        if isinstance(pred_cm, (tuple, list)): pred_cm = pred_cm[0]
        print(f"[SANITY@init] pred_cm: min={pred_cm.min():.2f}, max={pred_cm.max():.2f}, mean={pred_cm.mean():.2f}")

    # -------- TRAIN LOOP --------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        running = 0.0

        for i, (rgb, depth_cm) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            rgb, depth_cm = rgb.to(device), depth_cm.to(device)
            pred_cm = model(rgb)
            if isinstance(pred_cm, (tuple, list)): pred_cm = pred_cm[0]

            if pred_cm.shape != depth_cm.shape:
                depth_cm = F.interpolate(depth_cm, size=pred_cm.shape[-2:], mode="bilinear", align_corners=False)
            depth_cm = torch.clamp(depth_cm, MIN_CM, MAX_CM)

            ld, lssim, lnorm, lgrad = criterion(pred_cm, depth_cm)
            loss = ld + lssim + lnorm + lgrad
            (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running += loss.item()

        avg_loss = running / len(train_loader)


        # --- VALIDAZIONE ---
        model.eval()
        rmseL, relL, d1L = [], [], []
        with torch.no_grad():
            for rgb, depth_cm in val_loader:
                rgb, depth_cm = rgb.to(device), depth_cm.to(device)
                pred_cm = model(rgb)
                if isinstance(pred_cm, (tuple, list)): pred_cm = pred_cm[0]
                if pred_cm.shape != depth_cm.shape:
                    depth_cm = F.interpolate(depth_cm, size=pred_cm.shape[-2:], mode="bilinear", align_corners=False)
                rmse, rel, d1 = compute_metrics_m(to_meters(pred_cm), to_meters(depth_cm))
                rmseL.append(rmse); relL.append(rel); d1L.append(d1)

        rmse, rel, d1 = np.nanmean(rmseL), np.nanmean(relL), np.nanmean(d1L)
        print(f"Epoch {epoch}: Loss={avg_loss:.3f} | RMSE={rmse:.3f} m | REL={rel:.3f} | δ1={d1:.3f}")

        # === Update scheduler (DOPO la validazione!) ===
        if USE_DYNAMIC_LR:
            scheduler.step(rel)          # usa la metrica di validazione
        else:
            scheduler.step()             # StepLR classico

        # mostra sempre il LR corrente
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[SCHEDULER] LR attuale: {current_lr:.2e}")

        # --- LOG METRICS PER CURVE ---
        loss_curve.append(avg_loss)
        rel_curve.append(rel)
        lr_curve.append(current_lr)

        # --- DEBUG range ---
        with torch.no_grad():
            rgb, depth_cm = next(iter(val_loader))
            rgb, depth_cm = rgb.to(device), depth_cm.to(device)
            pred_cm = model(rgb)
            if isinstance(pred_cm, (tuple, list)): pred_cm = pred_cm[0]
            mn, mx, av = pred_cm.min().item(), pred_cm.max().item(), pred_cm.mean().item()
            print(f"[DEBUG] pred_cm range (cm): min={mn:.2f}, max={mx:.2f}, mean={av:.2f}")
            if abs(mx - mn) < 1e-2:
                print("⚠️ [WARN] Depth pred. quasi costante — possibile collasso.")


        # --- SAVE BEST/LAST ---
        if rel < best_rel:
            best_rel = rel
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{EXP_NAME}_best.pth"))
            print(f"[SAVE] ✅ Migliore AbsRel={best_rel:.3f}")
        else:
            no_improve_epochs += 1
            print(f"[INFO] REL non migliora da {no_improve_epochs} epoche ({rel:.3f} vs best {best_rel:.3f})")

        # early stop per stagnazione
        if no_improve_epochs >= early_stop_patience:
            print(f"🛑 Early stopping: nessun miglioramento REL per {early_stop_patience} epoche consecutive.")
            break

        # stop per collasso
        if abs(mx - mn) < 1e-2:
            collapse_counter += 1
            print(f"⚠️ Depth collassata ({collapse_counter}/3)")
            if collapse_counter >= 3:
                print("🛑 Fermato per collasso del modello (output costante).")
                break
        else:
            collapse_counter = 0

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{EXP_NAME}_last.pth"))

        # --- PREVIEWS ---
        if PREVIEW_INTERVAL and (epoch % PREVIEW_INTERVAL == 0):
            save_val_previews(model, val_loader, device, epoch, PREVIEW_DIR, PREVIEW_SAMPLES)

   
    # --- LOG CURVE ---
    # === Salva curva di training ===
    plt.figure(figsize=(10,6))
    epochs_range = range(1, len(loss_curve)+1)

    plt.subplot(2,1,1)
    plt.plot(epochs_range, loss_curve, label='Train Loss')
    plt.plot(epochs_range, rel_curve, label='Validation AbsRel')
    plt.legend(); plt.title('Training curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss / AbsRel')

    plt.subplot(2,1,2)
    plt.plot(epochs_range, lr_curve, color='orange', label='Learning Rate')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('LR')

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{EXP_NAME}_curves.png"))
    plt.close()
    print(f"[PLOT] 📈 Saved training curves → {os.path.join(OUT_DIR, f'{EXP_NAME}_curves.png')}")       
        

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train_meter()
