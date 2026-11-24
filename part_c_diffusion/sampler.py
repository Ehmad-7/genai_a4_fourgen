"""Part C: DDPM sampling loop (Algorithm 2), importable by sample.py and by the web app."""
import torch
from part_c_diffusion.schedule import make_schedule, T


@torch.no_grad()
def p_sample_loop(model, labels, device, seed=0, track_idx=None, track_ts=(999, 750, 500, 250, 100)):
    """x_T ~ N(0,I); for t = T..1:  x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
    + sigma_t * z,  sigma_t^2 = beta_t,  z = 0 at the last step. Final result clipped to [-1, 1]."""
    betas, alphas, abar = make_schedule(device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    B = len(labels)
    x = torch.randn(B, 1, 28, 28, device=device, generator=g)
    traj = {}
    for t in range(T - 1, -1, -1):
        if track_idx is not None and t in track_ts:
            traj[t] = x[track_idx].detach().cpu()
        tt = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(x, tt, class_labels=labels).sample
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn(x.shape, device=device, generator=g)
    x = x.clamp(-1, 1)
    if track_idx is not None:
        traj[0] = x[track_idx].detach().cpu()
    return x, traj


def to_img(t):
    return ((t + 1) / 2).clamp(0, 1)
