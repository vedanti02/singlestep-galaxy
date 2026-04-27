"""Global environment encoder.

Consumes the *stitched* low-fidelity cube — a coarse view of the entire
simulation that surrounds the current crop — and returns a single vector
``g_env`` per sample. ``g_env`` is then fused with a sinusoidal time
embedding and the cosmology vector to form the conditioning token used
by the per-point trunk.

The encoder is intentionally small (a few strided ConvBlocks → global
average pool → linear) because the env grid is only 64³ and is the same
for every crop of a given simulation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock3D, StyleTimeMLP


class GlobalContextEncoder(nn.Module):
    """3D-CNN env encoder + style/time fusion.

    Args:
        c_env_in: Channel count of the stitched env cube (3 for disp-only).
        n_style: Length of the cosmology vector.
        base: Width of the first conv block.
        cond_dim: Dimension of the output conditioning token.
        env_resolution: Side of the input env grid (default 64).
    """

    def __init__(self,
                 c_env_in: int = 3,
                 n_style: int = 5,
                 base: int = 32,
                 cond_dim: int = 256,
                 env_resolution: int = 64) -> None:
        super().__init__()
        self.env_resolution = env_resolution
        self.net = nn.Sequential(
            ConvBlock3D(c_env_in, base),                         # 64
            ConvBlock3D(base, base * 2, stride=2),               # 32
            ConvBlock3D(base * 2, base * 4, stride=2),           # 16
            ConvBlock3D(base * 4, base * 4, stride=2),           # 8
        )
        self.proj = nn.Linear(base * 4, cond_dim)
        self.style_time = StyleTimeMLP(n_style=n_style, c_out=cond_dim)
        self.cond_dim = cond_dim

    def encode_env(self, env: torch.Tensor) -> torch.Tensor:
        """Compute the env-only token g_env. Cached per batch in inference."""
        h = self.net(env)
        h = F.adaptive_avg_pool3d(h, 1).flatten(1)
        return self.proj(h)                                      # (B, cond_dim)

    def forward(self, env: torch.Tensor, style: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Return the fused conditioning token.

        Args:
            env:   ``(B, C, R, R, R)`` stitched LF env cube (R=env_resolution).
            style: ``(B, n_style)``    cosmology vector.
            t:     ``(B,)``            flow-matching time in ``[0, 1]``.

        Returns:
            ``(B, cond_dim)`` token.
        """
        g_env = self.encode_env(env)
        c_st  = self.style_time(t, style)
        return g_env + c_st
