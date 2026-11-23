"""Part C: DDPM noise schedule and the closed-form forward process.
Indexing: array index t = 0..T-1 is the time step fed to the U-Net (math step t+1)."""
import torch

T = 1000


def make_schedule(T=T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Linear betas, alphas = 1 - betas, alpha_bar = cumulative product of alphas."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas.float().to(device), alphas.float().to(device), alphas_bar.float().to(device)


def q_sample(x0, t, eps, alphas_bar):
    """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps   (x0 in [-1, 1])."""
    a = alphas_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * eps
