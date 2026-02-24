from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class CVAEConfig:
    image_size: int = 64
    num_classes: int = 26
    latent_dim: int = 32
    label_embed_dim: int = 16
    base_channels: int = 32

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for python_ai.model")


class ConditionalVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, config: CVAEConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        if config.image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")

        c = int(config.base_channels)
        self.label_embed = nn.Embedding(config.num_classes, config.label_embed_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        feat_hw = config.image_size // 16
        self._feat_hw = feat_hw
        self._enc_flat = (c * 8) * feat_hw * feat_hw
        hidden = max(128, c * 8)

        self.fc_enc = nn.Linear(self._enc_flat + config.label_embed_dim, hidden)
        self.fc_mu = nn.Linear(hidden, config.latent_dim)
        self.fc_logvar = nn.Linear(hidden, config.latent_dim)

        self.fc_dec = nn.Linear(config.latent_dim + config.label_embed_dim, self._enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c // 2 if c >= 16 else c, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        out_c = c // 2 if c >= 16 else c
        self.head = nn.Conv2d(out_c, 1, 3, 1, 1)

    def encode(self, x, labels):
        h = self.enc(x)
        h = h.view(h.shape[0], -1)
        y = self.label_embed(labels)
        h = torch.cat([h, y], dim=1)
        h = torch.relu(self.fc_enc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        y = self.label_embed(labels)
        h = torch.cat([z, y], dim=1)
        h = self.fc_dec(h)
        c = self.config.base_channels
        h = h.view(h.shape[0], c * 8, self._feat_hw, self._feat_hw)
        h = self.dec(h)
        return torch.sigmoid(self.head(h))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, labels, z=None):
        if z is None:
            z = torch.randn(labels.shape[0], self.config.latent_dim, device=labels.device)
        return self.decode(z, labels)


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    if F is None:
        raise RuntimeError("PyTorch is required for python_ai.model.vae_loss")
    recon = F.binary_cross_entropy(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + (float(beta) * kld)
    return {"loss": loss, "recon": recon, "kld": kld}
