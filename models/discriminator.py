"""3-D PatchGAN critics for adversarial training of the PVFM generator.

Two instances of :class:`PatchCritic3D` are used (see engine.trainer):

* ``D_disp`` — 6 input channels: predicted-or-real HF−LF residual
  displacement cube (3) ⊕ LF displacement cube (3, conditioning).
* ``D_dens`` — 2 input channels: ``log(rho + eps)`` of the
  predicted-or-real HF field (1) ⊕ same for the LF field (1,
  conditioning).

Both operate on the inner-cropped cube (edge buffer stripped) so the
critic never sees crop-boundary artifacts. Conditioning on LF makes the
critic judge the *pair* — a generator cannot win with HF-looking output
that is incoherent with its LF input, which protects the cross-coherence
metric r(k).

Design notes (WGAN-GP): no normalization layers (BatchNorm is invalid
with a per-sample gradient penalty; the nets are small enough to train
bare), LeakyReLU(0.2), strided convs down to a 3^3 patch-score grid whose
mean is the scalar critic output. Optional projection-style cosmology
conditioning (`style_dim > 0`) adds <style_embed, pooled_features> to the
score (Miyato & Koyama 2018), off by default.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PatchCritic3D(nn.Module):
    """Small 3-D convolutional critic returning one scalar per sample.

    Args:
        c_in: Input channels (field + conditioning channels, concatenated).
        base: Width of the first conv layer (doubles twice).
        style_dim: If > 0, enable the projection head on a style vector
            passed to :meth:`forward` (cosmology conditioning).
    """

    def __init__(self, c_in: int, base: int = 64, style_dim: int = 0) -> None:
        super().__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        c1, c2, c3 = base, base * 2, base * 4
        self.features = nn.Sequential(
            nn.Conv3d(c_in, c1, 3, stride=1, padding=1), act,   # 24 -> 24
            nn.Conv3d(c1, c2, 4, stride=2, padding=1), act,     # 24 -> 12
            nn.Conv3d(c2, c3, 4, stride=2, padding=1), act,     # 12 -> 6
            nn.Conv3d(c3, c3, 4, stride=2, padding=1), act,     #  6 -> 3
        )
        self.head = nn.Conv3d(c3, 1, 3, stride=1, padding=1)    # (B, 1, 3, 3, 3)
        self.style_dim = style_dim
        if style_dim > 0:
            self.style_proj = nn.Linear(style_dim, c3, bias=False)

    def forward(self, x: torch.Tensor,
                style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score a batch of conditioned field cubes.

        Args:
            x: ``(B, c_in, S, S, S)`` input (S = inner crop side, e.g. 24).
            style: ``(B, style_dim)`` cosmology vector (only if
                ``style_dim > 0``).

        Returns:
            ``(B,)`` unbounded critic scores (higher = "more real").
        """
        h = self.features(x)                                     # (B, C, 3, 3, 3)
        score = self.head(h).mean(dim=(1, 2, 3, 4))              # (B,)
        if self.style_dim > 0:
            if style is None:
                raise ValueError("style tensor required when style_dim > 0")
            pooled = h.mean(dim=(2, 3, 4))                       # (B, C)
            score = score + (self.style_proj(style) * pooled).sum(dim=1)
        return score
