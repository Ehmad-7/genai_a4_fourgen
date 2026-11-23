"""Part B correctness checks (no dataset needed): KL >= 0, KL = 0 at mu = 0 and log sigma^2 = 0."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.distributions import Normal, kl_divergence as torch_kl
from part_b_vae.model import VAE, kl_divergence, elbo_terms

torch.manual_seed(0)
model = VAE()
print("VAE parameters:", sum(p.numel() for p in model.parameters()))

# 1) KL is never negative (many random mu / log sigma^2, including extreme values)
mu = torch.randn(10000, 16) * 3
logvar = torch.randn(10000, 16) * 2
kl = kl_divergence(mu, logvar)
assert (kl >= 0).all(), "KL must never be negative"
print(f"OK: KL >= 0 for 10000 random posteriors (min {kl.min().item():.4f}, max {kl.max().item():.1f})")

# 2) KL = 0 exactly when mu = 0 and log sigma^2 = 0
kl0 = kl_divergence(torch.zeros(5, 16), torch.zeros(5, 16))
assert torch.all(kl0 == 0)
print("OK: KL = 0 when mu = 0 and log sigma^2 = 0 ->", kl0.tolist())

# 3) closed form agrees with PyTorch's own Normal-vs-Normal KL
std = torch.exp(0.5 * logvar)
ref = torch_kl(Normal(mu, std), Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(dim=1)
assert torch.allclose(kl, ref, rtol=1e-4, atol=1e-4)
print("OK: closed-form KL matches torch.distributions.kl_divergence")

# 4) reconstruction term = BCE summed over the 784 pixels (checked against the formula by hand)
x = torch.rand(8, 1, 28, 28)
logits, m, lv = model(x)
assert logits.shape == (8, 1, 28, 28) and m.shape == (8, 16) and lv.shape == (8, 16)
recon, _ = elbo_terms(logits, x, m, lv)
p = torch.sigmoid(logits.double())
manual = -(x.double() * torch.log(p) + (1 - x.double()) * torch.log(1 - p)).sum(dim=(1, 2, 3))
assert torch.allclose(recon.double(), manual, rtol=1e-4, atol=1e-3)
print("OK: reconstruction term = binary cross-entropy summed over 784 pixels; shapes (28x28 logits, 16-d mu/log-var)")
