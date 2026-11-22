"""Part A correctness checks (run locally, no dataset needed):
exactly 48 of 64 patches masked in every image, loss only on masked patches."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
from eurosat import patchify
from part_a_mae.model import MAE

torch.manual_seed(0)
model = MAE().eval()
print("MAE parameters:", sum(p.numel() for p in model.parameters()))

x = torch.rand(32, 3, 64, 64)
p = patchify(x)                                                   # (32, 64, 192)

# 1) exactly 48 masked / 16 visible per image, new random choice per image
ids_keep, mask = model.random_masking(32, p.device)
assert (mask.sum(dim=1) == 48).all(), "every image must have exactly 48 masked patches"
assert ids_keep.shape == (32, 16)
assert len({tuple(sorted(r.tolist())) for r in ids_keep}) > 1, "masks must differ between images"
print("OK: every image has exactly 48 of 64 patches masked (16 visible)")

# 2) loss is computed on masked patches only
with torch.no_grad():
    pred = model.decode(model.encode(p, ids_keep), ids_keep)
    loss = MAE.masked_loss(pred, p, mask)
    manual = torch.stack([((pred[i] - p[i]) ** 2).mean(-1)[mask[i] == 1].mean() for i in range(32)]).mean()
    p_changed = p.clone()
    p_changed[mask == 0] += 5.0                                   # corrupt VISIBLE targets only
    loss_changed = MAE.masked_loss(pred, p_changed, mask)
assert torch.allclose(loss, manual, atol=1e-6)
assert torch.allclose(loss, loss_changed), "loss must not depend on visible patches"
print(f"OK: loss = {loss.item():.4f} uses masked patches only (visible patches do not affect it)")

# 3) shapes
l, pr, m = model(p)
assert pr.shape == (32, 64, 192) and m.shape == (32, 64)
assert model.features(p).shape == (32, 192)
print("OK: shapes (pred, mask, features) =", tuple(pr.shape), tuple(m.shape), tuple(model.features(p).shape))
