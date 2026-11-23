"""Part A: reconstructions of 8 test tiles (masked input / reconstruction / original)
+ masked-patch MSE on the whole test set.  Run from repo root: python part_a_mae/reconstruct.py"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eurosat import load_eurosat_split, patchify, unpatchify, CLASS_NAMES
from part_a_mae.model import MAE

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MAE().to(device)
model.load_state_dict(torch.load(ROOT / "weights" / "mae.pt", map_location=device))
model.eval()

x_te, y_te = load_eurosat_split("test")

# masked-patch MSE over the whole test set (random 75% mask per image)
tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(x_te), 512):
        p = patchify(x_te[i : i + 512].to(device))
        loss, _, _ = model(p)
        tot += loss.item() * len(p)
        n += len(p)
test_mse = tot / n
print(f"test masked-patch MSE: {test_mse:.5f}")

# 8 test images, each from a different class
perm = torch.randperm(len(x_te), generator=torch.Generator().manual_seed(0)).tolist()
idx, seen = [], set()
for i in perm:
    c = int(y_te[i])
    if c not in seen:
        seen.add(c); idx.append(i)
    if len(idx) == 8:
        break

p = patchify(x_te[idx].to(device))                                    # (8, 64, 192)
with torch.no_grad():
    ids_keep, mask = model.random_masking(8, device)
    pred = model.decode(model.encode(p, ids_keep), ids_keep)
m = mask.unsqueeze(-1)                                                # 1 = masked
masked_in = p * (1 - m) + 0.5 * m                                     # masked patches grey
recon = p * (1 - m) + pred.clamp(0, 1) * m                            # predicted pasted into visible
rows = [unpatchify(t).cpu().clamp(0, 1) for t in (masked_in, recon, p)]

(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 8, figsize=(12.5, 5))
for r, (label, imgs) in enumerate(zip(["masked input", "reconstruction", "original"], rows)):
    for c in range(8):
        ax = axes[r, c]
        ax.imshow(imgs[c].permute(1, 2, 0).numpy())
        ax.axis("off")
        if r == 0:
            ax.set_title(CLASS_NAMES[int(y_te[idx[c]])], fontsize=7)
        if c == 0:
            ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=9)
fig.subplots_adjust(left=0.12, wspace=0.05, hspace=0.05)
fig.savefig(ROOT / "results" / "figures" / "mae_reconstructions.png", dpi=150, bbox_inches="tight")
json.dump({"test_masked_mse": test_mse, "test_indices": idx},
          open(ROOT / "results" / "mae_reconstruction_stats.json", "w"), indent=2)
print("saved results/figures/mae_reconstructions.png")
