"""Composable wrapper that ties the local + global encoders into one model.

This is intentionally a plain ``nn.Module`` rather than a
``LightningModule`` — keeping training-loop concerns out of the model
file makes the encoders reusable in other experiments (e.g. a
score-matching variant that swaps only :mod:`engine.flow_matching`).

A LightningModule wrapper can be added in :mod:`engine.trainer`
without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .global_context_encoder import GlobalContextEncoder
from .point_voxel_encoder import PointVoxelEncoder


@dataclass
class FlowMatcherInputs:
    """Bundle of all per-batch tensors the model consumes.

    This dataclass exists purely for documentation / type-clarity; nothing
    requires you to pass it (the ``forward`` method takes raw tensors so
    it composes cleanly with ``torch.compile`` and Lightning).
    """
    lf_voxel: torch.Tensor   # (B, c_lf, D, D, D)
    env: torch.Tensor        # (B, c_env, R, R, R)
    coords: torch.Tensor     # (B, N, 3)
    lf_pt: torch.Tensor      # (B, N, c_lf_pt)
    style: torch.Tensor      # (B, n_style)


class PVFlowMatcher(nn.Module):
    """Point-Voxel flow-matching model.

    Composition:
        - :class:`GlobalContextEncoder` produces a per-batch conditioning
          token from (env, style, t).
        - :class:`PointVoxelEncoder` runs the LF U-Net once per batch
          (cached via :meth:`encode_cond`) and predicts a velocity at
          every sampled point given ``(x_t, coords, lf_pt, lf_feat, cond)``.
    """

    def __init__(self,
                 c_pt: int = 3,
                 c_lf: int = 3,
                 c_env: int = 3,
                 c_lf_pt: int = 3,
                 n_style: int = 5,
                 base_voxel: int = 32,
                 base_point: int = 128,
                 cond_dim: int = 256,
                 n_blocks: int = 4,
                 env_resolution: int = 64) -> None:
        super().__init__()
        self.local = PointVoxelEncoder(
            c_pt=c_pt, c_lf=c_lf, c_lf_pt=c_lf_pt,
            base_voxel=base_voxel, base_point=base_point,
            cond_dim=cond_dim, n_blocks=n_blocks)
        self.globalc = GlobalContextEncoder(
            c_env_in=c_env, n_style=n_style, base=base_voxel,
            cond_dim=cond_dim, env_resolution=env_resolution)
        self.c_pt = c_pt

    def encode_cond(self,
                    lf_voxel: torch.Tensor,
                    env: torch.Tensor,
                    style: torch.Tensor,
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all conditioning encoders once per (batch, t) and cache results.

        Returns:
            Tuple ``(lf_feat, cond)``:

            * ``lf_feat`` — ``(B, C_v, D, D, D)`` LF voxel U-Net features.
            * ``cond``    — ``(B, cond_dim)`` global conditioning token.
        """
        lf_feat = self.local.encode_voxel(lf_voxel)
        cond    = self.globalc(env, style, t)
        return lf_feat, cond

    def forward(self,
                x_t: torch.Tensor,
                coords: torch.Tensor,
                lf_pt: torch.Tensor,
                lf_feat: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """Predict the flow-matching velocity at each point.

        Args:
            x_t:     ``(B, N, c_pt)`` interpolated state at time ``t``.
            coords:  ``(B, N, 3)``    normalized point coords in [0, 1].
            lf_pt:   ``(B, N, c_lf_pt)`` LF disp gathered at the points.
            lf_feat: ``(B, C_v, D, D, D)`` from :meth:`encode_cond`.
            cond:    ``(B, cond_dim)``    from :meth:`encode_cond`.

        Returns:
            ``(B, N, c_pt)`` velocity prediction.
        """
        return self.local(x_t, coords, lf_pt, lf_feat, cond)
