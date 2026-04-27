"""Reusable network primitives shared across encoders.

Kept tiny on purpose — the per-component complexity belongs to the
specialized encoders (:mod:`models.point_voxel_encoder`,
:mod:`models.global_context_encoder`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Safe GroupNorm: groups = min(max_groups, channels)."""
    return nn.GroupNorm(num_groups=min(max_groups, channels),
                        num_channels=channels)


class ConvBlock3D(nn.Module):
    """Two-conv ResBlock with GroupNorm + SiLU.

    Args:
        c_in: Input channel count.
        c_out: Output channel count.
        stride: Stride of the first conv (use 2 for downsampling).
        kernel_size: Kernel side (default 3, 'same' padding).
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1,
                 kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size, stride=stride, padding=pad)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size, padding=pad)
        self.n1 = group_norm(c_out)
        self.n2 = group_norm(c_out)
        self.skip = (nn.Conv3d(c_in, c_out, 1, stride=stride)
                     if (c_in != c_out or stride != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        return F.silu(h + self.skip(x))


def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Standard sinusoidal positional embedding for a scalar (e.g. flow time).

    Args:
        t: ``(B,)`` tensor of scalars in any range (typically ``[0, 1]``).
        dim: Output embedding dimension; must be even.

    Returns:
        ``(B, dim)`` tensor.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMPointMLP(nn.Module):
    """Per-point MLP with FiLM (scale/shift) conditioning from a global token.

    Implements ``y = silu( norm( fc2(silu(norm(fc1(x)))) ) * (1+s) + b ) + skip``
    where ``(s, b) = linear(cond).chunk(2)``. Used as the trunk block of
    the velocity-predicting head — the FiLM gate lets the global env+style+t
    token modulate every point.
    """

    def __init__(self, c_in: int, c_out: int, c_cond: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(c_in, c_out, 1)
        self.fc2 = nn.Conv1d(c_out, c_out, 1)
        self.n1 = group_norm(c_out)
        self.n2 = group_norm(c_out)
        self.cond = nn.Linear(c_cond, 2 * c_out)
        self.skip = (nn.Conv1d(c_in, c_out, 1)
                     if c_in != c_out else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N); cond: (B, c_cond)
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = F.silu(self.n1(self.fc1(x)))
        h = self.n2(self.fc2(h))
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return F.silu(h + self.skip(x))


class StyleTimeMLP(nn.Module):
    """Concatenate sinusoidal(t) ⊕ cosmology vector → conditioning token."""

    def __init__(self, n_style: int, c_out: int, t_dim: int = 128) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(t_dim + n_style, c_out),
            nn.SiLU(),
            nn.Linear(c_out, c_out),
        )

    def forward(self, t: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        e = sinusoidal_embedding(t, self.t_dim)
        return self.net(torch.cat([e, style], dim=-1))
