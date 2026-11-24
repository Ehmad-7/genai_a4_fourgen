"""Part C: DDPM sampling (Algorithm 2).
  python part_c_diffusion/sample.py            -> 40 x 8 grid (name on each row) + denoising of one sample
  python part_c_diffusion/sample.py --blind    -> one shuffled sample per character for the blind test"""
import argparse, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uhat import _find_split_dir
from part_c_diffusion.model import build_unet


from part_c_diffusion.sampler import p_sample_loop, to_img


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "weights" / "ddpm.pt"))
ap.add_argument("--classes", type=int, default=40)
ap.add_argument("--per_class", type=int, default=8)
ap.add_argument("--tag", default="")
ap.add_argument("--blind", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
uhat_root = str(ROOT / "starter" / "data" / "uhat")
names = sorted(c.name.lower() for c in _find_split_dir(uhat_root, "char", "train").iterdir() if c.is_dir())
model = build_unet().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

if args.blind:
    out_dir = ROOT / "results" / "blind_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = torch.arange(40, device=device)                       # one sample per character
    x, _ = p_sample_loop(model, labels, device, seed=args.seed + 1)
    order = np.random.default_rng(args.seed).permutation(40)      # position i shows class order[i]
    fig, axes = plt.subplots(5, 8, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_img(x[order[i], 0]).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"#{i + 1}", fontsize=11); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "blind_sheet.png", dpi=130)
    with open(out_dir / "_key_do_not_open.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "true_class"])
        for i in range(40):
            w.writerow([i + 1, names[order[i]]])
    with open(out_dir / "guesses.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "guess"])
        for i in range(40):
            w.writerow([i + 1, ""])
    (out_dir / "class_list.txt").write_text("\n".join(names) + "\n")
    print("saved results/blind_test/: blind_sheet.png, guesses.csv, class_list.txt, _key_do_not_open.csv")
    sys.exit(0)

# ---- 40 x 8 grid: 8 samples for each character ----
nc, k = args.classes, args.per_class
labels = torch.arange(nc, device=device).repeat_interleave(k)       # 0,0,..,1,1,..
x, traj = p_sample_loop(model, labels, device, seed=args.seed, track_idx=0)
torch.save(x.cpu(), ROOT / "results" / f"{args.tag}ddpm_samples.pt")
x = to_img(x).cpu()
fig, axes = plt.subplots(nc, k, figsize=(k * 0.9 + 1.2, nc * 0.9), squeeze=False)
for r in range(nc):
    for c in range(k):
        ax = axes[r, c]
        ax.imshow(x[r * k + c, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
    axes[r, 0].text(-0.15, 0.5, names[r], transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
fig.subplots_adjust(left=0.2, wspace=0.05, hspace=0.05)
fig.savefig(fig_dir / f"{args.tag}ddpm_samples_grid.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---- denoising of one sample (first sample of the first character) ----
ts = [999, 750, 500, 250, 100, 0]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
for ax, t in zip(axes, ts):
    ax.imshow(to_img(traj[t][0]).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t = {t}", fontsize=9); ax.axis("off")
fig.suptitle(f"denoising one sample ('{names[0]}')", fontsize=9)
fig.tight_layout()
fig.savefig(fig_dir / f"{args.tag}ddpm_denoising.png", dpi=150)
print(f"saved {args.tag}ddpm_samples_grid.png and {args.tag}ddpm_denoising.png in results/figures/")
