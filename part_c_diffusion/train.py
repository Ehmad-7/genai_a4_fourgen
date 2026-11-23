"""Part C: train the class-conditional DDPM (Algorithm 1, L_simple).
Adam lr 2e-4, batch 128, 40 epochs, pixels rescaled to [-1, 1]. Run from repo root: python part_c_diffusion/train.py"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uhat import load_uhat_characters
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import make_schedule, q_sample, T

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
args = ap.parse_args()

prefix = "smoke_" if args.limit else ""
(ROOT / "weights").mkdir(exist_ok=True)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print("device:", device, gpu)

x_tr, y_tr, names = load_uhat_characters(root=str(ROOT / "starter" / "data" / "uhat"), split="train")
if args.limit:
    x_tr, y_tr = x_tr[: args.limit], y_tr[: args.limit]
x = (x_tr * 2 - 1).to(device)                          # [0, 1] -> [-1, 1]
y = y_tr.to(device)
N = len(x)
print("training images:", N, "| classes:", len(names))

model = build_unet().to(device)
n_params = sum(p.numel() for p in model.parameters())
print("U-Net parameters:", n_params)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
betas, alphas, abar = make_schedule(device=device)

log, t0 = [], time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(N, device=device)
    total = 0.0
    for i in range(0, N, args.bs):
        idx = perm[i : i + args.bs]
        x0, yb = x[idx], y[idx]
        t = torch.randint(0, T, (len(idx),), device=device)          # uniform over the T steps
        eps = torch.randn_like(x0)
        xt = q_sample(x0, t, eps, abar)
        loss = F.mse_loss(model(xt, t, class_labels=yb).sample, eps)  # L_simple
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    log.append((epoch, total / N))
    print(f"epoch {epoch:2d}/{args.epochs}  loss {total / N:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    if epoch % 5 == 0 or epoch == args.epochs:
        torch.save(model.state_dict(), ROOT / "weights" / f"{prefix}ddpm.pt")

elapsed = time.time() - t0
with open(ROOT / "results" / f"{prefix}ddpm_loss.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "loss"]); w.writerows(log)
json.dump({"params": n_params, "final_loss": log[-1][1], "seconds": elapsed, "device": gpu, "epochs": args.epochs},
          open(ROOT / "results" / f"{prefix}ddpm_training.json", "w"), indent=2)
plt.figure(figsize=(5, 3.5))
plt.plot([e for e, _ in log], [l for _, l in log])
plt.xlabel("epoch"); plt.ylabel("L_simple (MSE on noise)"); plt.title("DDPM training loss")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(ROOT / "results" / "figures" / f"{prefix}ddpm_loss.png", dpi=150)
print(f"done in {elapsed:.0f}s; final loss {log[-1][1]:.5f}")
