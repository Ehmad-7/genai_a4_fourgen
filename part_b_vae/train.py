"""Part B: train the VAE on UHaT characters (labels unused). Adam lr 1e-3, batch 128, 30 epochs.
Logs reconstruction term, KL term and their sum (negative ELBO, nats/image) per epoch, train and test.
Run from repo root:  python part_b_vae/train.py"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uhat import load_uhat_characters
from part_b_vae.model import VAE, elbo_terms

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--limit", type=int, default=0, help="use only N train images (smoke test)")
args = ap.parse_args()

prefix = "smoke_" if args.limit else ""
(ROOT / "weights").mkdir(exist_ok=True)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print("device:", device, gpu)

uhat_root = str(ROOT / "starter" / "data" / "uhat")
x_tr, _, names = load_uhat_characters(root=uhat_root, split="train")
x_te, _, _ = load_uhat_characters(root=uhat_root, split="test", class_names=names)
if args.limit:
    x_tr, x_te = x_tr[: args.limit], x_te[: args.limit]
x_tr, x_te = x_tr.to(device), x_te.to(device)
print("train", tuple(x_tr.shape), "test", tuple(x_te.shape))


@torch.no_grad()
def evaluate(model, x, bs=1000):
    """Mean recon / KL per image on x; fixed noise so numbers are comparable across epochs."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    r_sum = k_sum = 0.0
    for i in range(0, len(x), bs):
        xb = x[i : i + bs]
        eps = torch.randn(len(xb), model.latent_dim, generator=g).to(device)
        logits, mu, logvar = model(xb, eps)
        r, k = elbo_terms(logits, xb, mu, logvar)
        r_sum += r.sum().item(); k_sum += k.sum().item()
    return r_sum / len(x), k_sum / len(x)


model = VAE().to(device)
n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

rows, t0, N = [], time.time(), len(x_tr)
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(N, device=device)
    r_sum = k_sum = 0.0
    for i in range(0, N, args.bs):
        xb = x_tr[perm[i : i + args.bs]]
        logits, mu, logvar = model(xb)
        r, k = elbo_terms(logits, xb, mu, logvar)
        loss = (r + k).mean()                         # negative ELBO averaged over the batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        r_sum += r.sum().item(); k_sum += k.sum().item()
    tr_r, tr_k = r_sum / N, k_sum / N
    te_r, te_k = evaluate(model, x_te)
    rows.append([epoch, tr_r, tr_k, tr_r + tr_k, te_r, te_k, te_r + te_k])
    print(f"epoch {epoch:2d}/{args.epochs}  train recon {tr_r:7.2f} KL {tr_k:6.2f} total {tr_r + tr_k:7.2f} | "
          f"test recon {te_r:7.2f} KL {te_k:6.2f} total {te_r + te_k:7.2f}  ({time.time() - t0:.0f}s)", flush=True)

elapsed = time.time() - t0
torch.save(model.state_dict(), ROOT / "weights" / f"{prefix}vae.pt")
with open(ROOT / "results" / f"{prefix}vae_loss.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_recon", "train_kl", "train_total", "test_recon", "test_kl", "test_total"])
    w.writerows(rows)
last = rows[-1]
json.dump({"params": n_params, "final_train_total": last[3], "test_recon": last[4], "test_kl": last[5],
           "test_neg_elbo": last[6], "seconds": elapsed, "device": gpu, "epochs": args.epochs},
          open(ROOT / "results" / f"{prefix}vae_training.json", "w"), indent=2)

ep = [r[0] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (title, tr, te) in zip(axes, [("reconstruction term (BCE)", 1, 4), ("KL term", 2, 5),
                                      ("negative ELBO = recon + KL", 3, 6)]):
    ax.plot(ep, [r[tr] for r in rows], label="train")
    ax.plot(ep, [r[te] for r in rows], label="test")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("nats / image"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / f"{prefix}vae_loss_terms.png", dpi=150)
print(f"done in {elapsed:.0f}s; test negative ELBO {last[6]:.2f} nats (recon {last[4]:.2f} + KL {last[5]:.2f})")
