"""Part C correctness checks + forward-noising figure.
alpha_bar_T ~ 4e-5 (x_T is almost pure noise); noised image at t = 999 looks like noise.
Run from repo root: python part_c_diffusion/checks.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uhat import load_uhat_characters
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import make_schedule, q_sample, T

torch.manual_seed(0)
betas, alphas, abar = make_schedule()

# 1) schedule
assert len(betas) == T == 1000
assert abs(betas[0].item() - 1e-4) < 1e-9 and abs(betas[-1].item() - 0.02) < 1e-9
assert (abar[1:] < abar[:-1]).all(), "alpha_bar must decrease"
print(f"OK: T = {T}, beta from {betas[0].item():.0e} to {betas[-1].item():.2f} (linear)")
print(f"alpha_bar at t=0: {abar[0].item():.4f}, t=999 (alpha_bar_T): {abar[-1].item():.3e}")
assert 3e-5 < abar[-1].item() < 5e-5
print("OK: alpha_bar_T is about 4e-5, so x_T is almost pure noise")

# 2) x_999 really is noise: unit variance, ~no correlation with the clean image
uhat_root = str(ROOT / "starter" / "data" / "uhat")
x_tr, y_tr, names = load_uhat_characters(root=uhat_root, split="train")
big = x_tr[:2000] * 2 - 1                                     # rescale [0,1] -> [-1,1]
x999 = q_sample(big, torch.full((2000,), 999), torch.randn_like(big), abar)
corr = np.corrcoef(big.flatten().numpy(), x999.flatten().numpy())[0, 1]
print(f"x_999: mean {x999.mean().item():.3f}, std {x999.std().item():.3f}, corr with clean image {corr:.4f}")
assert abs(x999.std().item() - 1) < 0.02 and abs(corr) < 0.05
print("OK: x_999 has unit variance and is uncorrelated with the clean image (looks like noise)")

# 3) forward process figure: one training image at t in {0, 100, 250, 500, 750, 999}
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
x0 = (x_tr[torch.randperm(len(x_tr))[:1]] * 2 - 1)
eps = torch.randn_like(x0)
ts = [0, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
for ax, t in zip(axes, ts):
    xt = q_sample(x0, torch.tensor([t]), eps, abar)
    ax.imshow(((xt[0, 0] + 1) / 2).clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t = {t}", fontsize=9); ax.axis("off")
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "ddpm_forward_noising.png", dpi=150)
print("saved results/figures/ddpm_forward_noising.png")

# 4) U-Net builds and runs with class conditioning
unet = build_unet()
print("U-Net parameters:", sum(p.numel() for p in unet.parameters()))
out = unet(torch.randn(4, 1, 28, 28), torch.randint(0, T, (4,)), class_labels=torch.randint(0, 40, (4,))).sample
assert out.shape == (4, 1, 28, 28)
print("OK: U-Net(x_t, t, class_labels=y) ->", tuple(out.shape))
