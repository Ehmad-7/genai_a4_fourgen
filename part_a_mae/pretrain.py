"""Part A: MAE pretraining on the 16.2k EuroSAT train images (labels ignored).
AdamW, lr 5e-4, weight decay 0.05, batch 256, 100 epochs. Logs loss per epoch.
Run from the repo root:  python part_a_mae/pretrain.py"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eurosat import load_eurosat_split, patchify
from part_a_mae.model import MAE

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
args = ap.parse_args()

prefix = "smoke_" if args.limit else ""
(ROOT / "weights").mkdir(exist_ok=True)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print("device:", device, gpu)

x, _ = load_eurosat_split("train")            # labels are NOT used in pretraining
if args.limit:
    x = x[: args.limit]
x = x.to(device)
N = x.size(0)
print("training images:", N)

model = MAE().to(device)
n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

log, t0 = [], time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(N, device=device)
    total = 0.0
    for i in range(0, N, args.bs):
        idx = perm[i : i + args.bs]
        loss, _, _ = model(patchify(x[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    ep_loss = total / N
    log.append((epoch, ep_loss))
    print(f"epoch {epoch:3d}/{args.epochs}  loss {ep_loss:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    if epoch % 10 == 0 or epoch == args.epochs:
        torch.save(model.state_dict(), ROOT / "weights" / f"{prefix}mae.pt")

elapsed = time.time() - t0
with open(ROOT / "results" / f"{prefix}mae_loss.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_mse_masked"]); w.writerows(log)
json.dump({"params": n_params, "final_loss": log[-1][1], "seconds": elapsed, "device": gpu,
           "epochs": args.epochs, "train_images": N},
          open(ROOT / "results" / f"{prefix}mae_training.json", "w"), indent=2)

plt.figure(figsize=(5, 3.5))
plt.plot([e for e, _ in log], [l for _, l in log])
plt.xlabel("epoch"); plt.ylabel("MSE on masked patches"); plt.title("MAE pretraining loss")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(ROOT / "results" / "figures" / f"{prefix}mae_pretrain_loss.png", dpi=150)
print(f"done in {elapsed:.0f}s; final loss {log[-1][1]:.5f}")
