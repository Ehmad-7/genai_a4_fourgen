"""Part A: Masked Autoencoder (MAE) for 64x64 EuroSAT tiles (64 patches of 8x8x3 = 192 numbers)."""
import torch
import torch.nn as nn


def _tf_layers(dim, heads, ff, depth):
    layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff,
                                       dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class MAE(nn.Module):
    def __init__(self, num_patches=64, patch_dim=192, enc_dim=192, enc_depth=6, enc_heads=3,
                 enc_ff=768, dec_dim=128, dec_depth=2, dec_heads=4, dec_ff=512, mask_ratio=0.75):
        super().__init__()
        self.num_patches = num_patches
        self.num_keep = int(num_patches * (1 - mask_ratio))          # 16 visible, 48 masked

        # ---- encoder (sees visible patches only) ----
        self.patch_embed = nn.Linear(patch_dim, enc_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, enc_dim))
        self.encoder = _tf_layers(enc_dim, enc_heads, enc_ff, enc_depth)
        self.enc_norm = nn.LayerNorm(enc_dim)

        # ---- decoder (visible tokens + mask tokens) ----
        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        self.decoder = _tf_layers(dec_dim, dec_heads, dec_ff, dec_depth)
        self.pred = nn.Linear(dec_dim, patch_dim)

        for p in (self.pos_embed, self.dec_pos, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)

    # a new random choice for every image and every call
    def random_masking(self, batch_size, device):
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_keep = noise.argsort(dim=1)[:, : self.num_keep]            # (B, 16)
        mask = torch.ones(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_keep, 0.0)                                 # 1 = masked, 0 = visible
        return ids_keep, mask

    def encode(self, patches, ids_keep=None):
        """patches (B, 64, 192). With ids_keep: only those patches go through the encoder."""
        x = self.patch_embed(patches) + self.pos_embed
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return self.enc_norm(self.encoder(x))

    def decode(self, enc, ids_keep):
        B = enc.size(0)
        d = self.dec_embed(enc)                                         # (B, 16, 128)
        full = self.mask_token.expand(B, self.num_patches, -1)          # mask token everywhere
        full = full.scatter(1, ids_keep.unsqueeze(-1).expand(-1, -1, d.size(-1)), d)
        full = self.decoder(full + self.dec_pos)
        return self.pred(full)                                          # (B, 64, 192)

    @staticmethod
    def masked_loss(pred, target, mask):
        """MSE on the masked patches only."""
        per_patch = ((pred - target) ** 2).mean(dim=-1)                 # (B, 64)
        return (per_patch * mask).sum() / mask.sum()

    def forward(self, patches):
        ids_keep, mask = self.random_masking(patches.size(0), patches.device)
        pred = self.decode(self.encode(patches, ids_keep), ids_keep)
        return self.masked_loss(pred, patches, mask), pred, mask

    @torch.no_grad()
    def features(self, patches):
        """No masking: all 64 patches -> encoder -> mean over patches -> (B, 192)."""
        return self.encode(patches).mean(dim=1)
