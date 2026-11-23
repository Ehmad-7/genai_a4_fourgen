"""Part B: Variational autoencoder for 28x28 UHaT characters (latent dimension 16, Bernoulli decoder)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        # encoder q(z|x): 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)        # log sigma^2
        # decoder p(x|z): 7x7 -> 14x14 -> 28x28, one logit per pixel
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, eps=None):
        """z = mu + sigma * eps, eps ~ N(0, I), sigma = exp(0.5 * log sigma^2)."""
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, 64, 7, 7))      # logits (B, 1, 28, 28)

    def forward(self, x, eps=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, eps)
        return self.decode(z), mu, logvar


def kl_divergence(mu, logvar):
    """Closed-form KL( N(mu, sigma^2) || N(0, I) ) per image, summed over the 16 latent dims."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def elbo_terms(logits, x, mu, logvar):
    """Per-image reconstruction term (BCE summed over 784 pixels) and KL term, both in nats."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))
    return recon, kl_divergence(mu, logvar)


def negative_elbo(logits, x, mu, logvar):
    """Negative ELBO averaged over the batch -> (total, recon, kl)."""
    recon, kl = elbo_terms(logits, x, mu, logvar)
    recon, kl = recon.mean(), kl.mean()
    return recon + kl, recon, kl
