"""Part A: few-label classifier on the FROZEN MAE encoder (160 labelled images, 16 per class).
Feature = mean of the 64 encoder outputs. nn.Linear(192, 10), AdamW lr 1e-3, 200 epochs, full batch.
Also reports the same head on a randomly initialised (not pretrained) encoder for comparison.
Run from repo root: python part_a_mae/classify.py"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from eurosat import load_eurosat_split, few_label_indices, patchify, CLASS_NAMES
from part_a_mae.model import MAE

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract(model, x, bs=512):
    out = []
    for i in range(0, len(x), bs):
        out.append(model.features(patchify(x[i : i + bs].to(device))).cpu())
    return torch.cat(out)


def train_head(f_tr, y_tr, f_te, y_te):
    torch.manual_seed(42)
    head = nn.Linear(192, 10)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for ep in range(1, 201):
        loss = F.cross_entropy(head(f_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  epoch {ep:3d}  train loss {loss.item():.4f}")
    with torch.no_grad():
        train_acc = (head(f_tr).argmax(1) == y_tr).float().mean().item()
        pred = head(f_te).argmax(1)
    return head, pred, (pred == y_te).float().mean().item(), train_acc


x_tr, y_tr = load_eurosat_split("train")
x_te, y_te = load_eurosat_split("test")
few = few_label_indices(y_tr)
x_few, y_few = x_tr[few], y_tr[few]
print("labelled images:", len(few), "| per class:", torch.bincount(y_few).tolist())

# ---- pretrained (frozen) encoder ----
model = MAE().to(device)
model.load_state_dict(torch.load(ROOT / "weights" / "mae.pt", map_location=device))
model.eval()
f_few, f_te = extract(model, x_few), extract(model, x_te)
print("pretrained encoder:")
head, pred, acc, train_acc = train_head(f_few, y_few, f_te, y_te)
print(f"TEST ACCURACY (pretrained MAE encoder): {acc*100:.2f}%   (train acc {train_acc*100:.1f}%, chance 10%)")

# ---- comparison: random, untrained encoder with the same head ----
torch.manual_seed(0)
rand_model = MAE().to(device).eval()
print("random-init encoder:")
_, _, acc_rand, _ = train_head(extract(rand_model, x_few), y_few, extract(rand_model, x_te), y_te)
print(f"TEST ACCURACY (random encoder): {acc_rand*100:.2f}%")

# ---- confusion matrix ----
cm = confusion_matrix(y_te.numpy(), pred.numpy(), labels=list(range(10)))
per_class = cm.diagonal() / cm.sum(1)
for name, a in zip(CLASS_NAMES, per_class):
    print(f"  {name:22s} {a*100:5.1f}%")
off = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(10) for j in range(10) if i != j]
print("most common confusions (count, true -> predicted):")
for c, t, pr in sorted(off, reverse=True)[:5]:
    print(f"  {c:4d}  {t} -> {pr}")

(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
np.savetxt(ROOT / "results" / "mae_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
           header=",".join(CLASS_NAMES), comments="")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"MAE few-label classifier, test acc {acc*100:.1f}%")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, fraction=0.046); fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "mae_confusion_matrix.png", dpi=150)

torch.save(head.state_dict(), ROOT / "weights" / "mae_classifier.pt")
json.dump({"test_accuracy": acc, "train_accuracy_160": train_acc, "random_encoder_test_accuracy": acc_rand,
           "per_class_accuracy": dict(zip(CLASS_NAMES, per_class.tolist()))},
          open(ROOT / "results" / "mae_classifier.json", "w"), indent=2)
print("saved confusion matrix, classifier head and json")
