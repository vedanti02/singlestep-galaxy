"""Point-Voxel encoder for the local LF crop.

Produces (a) a multi-scale voxel feature map of the LF input crop and
(b) the per-point trunk that consumes the noisy state ``x_t``,
sampled coordinates, and gathered voxel features. The trunk's final
output is a 3-vector per point (the flow-matching velocity).

This module is **decoupled** from the global env / style / time
encoding: the trunk takes a precomputed conditioning token. That makes
it possible to swap the env encoder out (e.g. for an attention-based
one) without touching this file.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ops.geometry import trilinear_devoxelize

from .blocks import ConvBlock3D, FiLMPointMLP, group_norm


class _LFVoxelUNet(nn.Module):
    """3-level 3D U-Net producing per-voxel features of the LF crop."""

    def __init__(self, c_in: int = 3, base: int = 32) -> None:
        super().__init__()
        self.stem  = ConvBlock3D(c_in, base)
        self.down1 = ConvBlock3D(base, base * 2, stride=2)
        self.down2 = ConvBlock3D(base * 2, base * 4, stride=2)
        self.bot   = ConvBlock3D(base * 4, base * 4)
        self.up2   = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2  = ConvBlock3D(base * 4, base * 2)
        self.up1   = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1  = ConvBlock3D(base * 2, base)
        self.out_channels = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        b  = self.bot(f2)
        u2 = self.dec2(torch.cat([self.up2(b), f1], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), f0], dim=1))
        return u1                                                # (B, base, D,D,D)


class PointVoxelEncoder(nn.Module):
    """Local Point-Voxel encoder + velocity head.

    Inputs (per batch):
        lf_voxel: ``(B, c_lf, D, D, D)`` LF voxel grid.
        coords:   ``(B, N, 3)``         normalized point coords in [0, 1].
        lf_pt:    ``(B, N, c_lf)``      LF disp gathered at the points.
        x_t:      ``(B, N, c_pt)``      the flow-matching state at the points.
        cond:     ``(B, c_cond)``       global conditioning token (env+style+time).

    Output:
        ``(B, N, c_pt)`` velocity prediction (same dim as ``x_t``).
    """

    def __init__(self,
                 c_pt: int = 3,                # x_t / output channels
                 c_lf: int = 3,                # LF voxel channels
                 c_lf_pt: int = 3,             # gathered LF channels
                 base_voxel: int = 32,
                 base_point: int = 128,
                 cond_dim: int = 256,
                 n_blocks: int = 4) -> None:
        super().__init__()
        self.lf_unet = _LFVoxelUNet(c_in=c_lf, base=base_voxel)
        c_v = self.lf_unet.out_channels                          # voxel feat dim

        # input concat: x_t ⊕ coords ⊕ LF_pt ⊕ voxel_feat_at_pt
        c_in = c_pt + 3 + c_lf_pt + c_v
        self.in_proj = nn.Conv1d(c_in, base_point, 1)

        self.trunk = nn.ModuleList([
            FiLMPointMLP(base_point, base_point, cond_dim)
            for _ in range(n_blocks)
        ])
        self.mid_voxel_proj = nn.Conv1d(c_v, base_point, 1)

        self.head = nn.Sequential(
            nn.Conv1d(base_point, base_point, 1),
            nn.SiLU(),
            nn.Conv1d(base_point, c_pt, 1),
        )
        self.c_v = c_v
        self.c_pt = c_pt

    def encode_voxel(self, lf_voxel: torch.Tensor) -> torch.Tensor:
        """Run the LF U-Net once per batch (cached by the FlowMatcher wrapper)."""
        return self.lf_unet(lf_voxel)

    def forward(self,
                x_t: torch.Tensor,
                coords: torch.Tensor,
                lf_pt: torch.Tensor,
                lf_feat: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        v_at_pt = trilinear_devoxelize(lf_feat, coords)            # (B, c_v, N)
        feat = torch.cat([
            x_t.transpose(1, 2),                                   # (B, c_pt, N)
            coords.transpose(1, 2),                                # (B, 3, N)
            lf_pt.transpose(1, 2),                                 # (B, c_lf_pt, N)
            v_at_pt,                                               # (B, c_v, N)
        ], dim=1)
        h = self.in_proj(feat)
        for i, blk in enumerate(self.trunk):
            h = blk(h, cond)
            if i == len(self.trunk) // 2:
                h = h + self.mid_voxel_proj(v_at_pt)
        return self.head(h).transpose(1, 2)                        # (B, N, c_pt)
