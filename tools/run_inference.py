import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from architecture import build_METER_model

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arch_type = "s"   # "xxs", "xs" o "s"
weights_path = "checkpoints_meter_cm/meter_s_best.pth"   # metti qui il path corretto
image_path = "/mnt/g/nyu_depth_v2/nyu_depth_v2/official_splits/test/bathroom/rgb_00510.jpg"

# === 1. Carica modello ===
model = build_METER_model(device=device, arch_type=arch_type).to(device)
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# === 2. Preprocessa immagine ===
transform = T.Compose([
    T.Resize((192, 256)),     # cambia se la risoluzione è diversa
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# === 3. Forward ===
with torch.no_grad():
    pred = model(input_tensor)
    depth_map = pred.squeeze().cpu().numpy()

# === 4. Salva/mostra output ===
plt.imshow(depth_map, cmap="plasma")
plt.colorbar()
plt.savefig("output_depth.png")
print("Salvata depth map in output_depth.png")
