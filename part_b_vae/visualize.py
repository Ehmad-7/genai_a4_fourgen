"""Part B figures: (a) 16 test images + reconstructions, (b) 64 samples from N(0, I),
(c) interpolations between mu of two test images of different characters (10 equally spaced points).
Run from repo root: python part_b_vae/visualize.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uhat import load_uhat_characters, _find_split_dir
from viz import save_grid
from part_b_vae.model import VAE

uhat_root = str(ROOT / "starter" / "data" / "uhat")
names = sorted(c.name.lower() for c in _find_split_dir(uhat_root, "char", "train").iterdir() if c.is_dir())
x_te, y_te, _ = load_uhat_characters(root=uhat_root, split="test", class_names=names)

model = VAE()
model.load_state_dict(torch.load(ROOT / "weights" / "vae.pt", map_location="cpu"))
model.eval()
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(x_te), generator=g)

# (a) 16 test images and their reconstructions (decoded from the posterior mean mu)
xb = x_te[perm[:16]]
with torch.no_grad():
    mu, _ = model.encode(xb)
    rec = torch.sigmoid(model.decode(mu))
fig, axes = plt.subplots(2, 16, figsize=(16, 2.4))
for r, (label, imgs) in enumerate([("original", xb), ("reconstruction", rec)]):
    for c in range(16):
        axes[r, c].imshow(imgs[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].text(-0.1, 0.5, label, transform=axes[r, c].transAxes, ha="right", va="center", fontsize=9)
fig.subplots_adjust(left=0.1, wspace=0.05, hspace=0.05)
fig.savefig(fig_dir / "vae_reconstructions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# (b) 64 new characters from z ~ N(0, I)
with torch.no_grad():
    samples = torch.sigmoid(model.decode(torch.randn(64, model.latent_dim, generator=g)))
save_grid(samples, str(fig_dir / "vae_samples.png"), nrow=8)

# (c) interpolation: 3 pairs, each of two DIFFERENT characters; 10 equally spaced points on the mu-line
picked, seen = [], set()
for i in perm.tolist():
    if int(y_te[i]) not in seen:
        seen.add(int(y_te[i])); picked.append(i)
    if len(picked) == 6:
        break
ts = torch.linspace(0, 1, 10).unsqueeze(1)
fig, axes = plt.subplots(3, 12, figsize=(13, 3.9))
for row in range(3):
    a, b = picked[2 * row], picked[2 * row + 1]
    with torch.no_grad():
        mu_a, _ = model.encode(x_te[a : a + 1])
        mu_b, _ = model.encode(x_te[b : b + 1])
        z = (1 - ts) * mu_a + ts * mu_b                                 # (10, 16)
        dec = torch.sigmoid(model.decode(z))
    cells = [x_te[a]] + [d for d in dec] + [x_te[b]]
    for c, img in enumerate(cells):
        ax = axes[row, c]
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes[row, 0].set_title(names[int(y_te[a])], fontsize=7)
    axes[row, 11].set_title(names[int(y_te[b])], fontsize=7)
fig.subplots_adjust(wspace=0.05, hspace=0.25)
fig.savefig(fig_dir / "vae_interpolation.png", dpi=150, bbox_inches="tight")
print("saved vae_reconstructions.png, vae_samples.png, vae_interpolation.png in results/figures/")
