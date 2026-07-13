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
                 env_resolution: int = 64,
                 z_dim: int = 0) -> None:
        super().__init__()
        self.local = PointVoxelEncoder(
            c_pt=c_pt, c_lf=c_lf, c_lf_pt=c_lf_pt,
            base_voxel=base_voxel, base_point=base_point,
            cond_dim=cond_dim, n_blocks=n_blocks)
        self.globalc = GlobalContextEncoder(
            c_env_in=c_env, n_style=n_style, base=base_voxel,
            cond_dim=cond_dim, env_resolution=env_resolution)
        self.c_pt = c_pt
        # Optional latent for conditional stochasticity (GAN training):
        # a deterministic conditional generator cannot match the one-to-
        # many LF -> HF distribution at sub-LF scales, so a per-sample
        # z ~ N(0, I) is projected into the conditioning token. Sampling
        # happens inside encode_cond so every sampler (lf_init_sample,
        # euler_sample, eval scripts) picks it up without changes.
        self.z_dim = z_dim
        if z_dim > 0:
            self.z_proj = nn.Sequential(
                nn.Linear(z_dim, cond_dim), nn.SiLU(),
                nn.Linear(cond_dim, cond_dim))

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
        return lf_feat, self.cond_token(env, style, t)

    def cond_token(self,
                   env: torch.Tensor,
                   style: torch.Tensor,
                   t: torch.Tensor,
                   g_env: torch.Tensor | None = None) -> torch.Tensor:
        """Global conditioning token (incl. latent z when ``z_dim > 0``).

        Exposed separately from :meth:`encode_cond` so callers that need
        several tokens per batch (e.g. the trainer's random-t MSE forward
        plus the adversarial t=0 forward) can pay for the env CNN once:
        pass ``g_env = model.globalc.encode_env(env)`` and only the cheap
        style/time MLP (+ a fresh z sample) is recomputed.
        """
        if g_env is None:
            g_env = self.globalc.encode_env(env)
        cond = g_env + self.globalc.style_time(t, style)
        if self.z_dim > 0:
            z = torch.randn(cond.shape[0], self.z_dim,
                            device=cond.device, dtype=cond.dtype)
            cond = cond + self.z_proj(z)
        return cond

    @classmethod
    def from_config(cls, cfg: dict) -> "PVFlowMatcher":
        """Build from a run Config — the ONE place channel defaults live.

        Used by the trainer, eval_physical, and inference_distributed so
        the architecture reconstruction cannot drift between them (a
        previous hardcoded copy in inference broke multi-field and
        c_lf_pt=0 checkpoints).
        """
        m = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        default_c_env = 4 if data_cfg.get("env_outside_mask", True) else 3
        n_fields = len(data_cfg.get("fields", ["disp"]))
        return cls(
            c_pt=3,
            c_lf=m.get("c_lf", 3 * n_fields),
            c_env=m.get("c_env", default_c_env),
            c_lf_pt=m.get("c_lf_pt", 3 * n_fields),
            n_style=m.get("n_style", 5),
            base_voxel=m.get("base_voxel", 32),
            base_point=m.get("base_point", 128),
            cond_dim=m.get("cond_dim", 256),
            n_blocks=m.get("n_blocks", 4),
            env_resolution=m.get("env_resolution", 64),
            z_dim=m.get("z_dim", 0),
        )

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
