"""Conditional flow-matching utilities (training targets + sampler).

Following Lipman et al. (2023). For a target sample :math:`x_1` and a
random reference :math:`x_0 \\sim \\mathcal{N}(0, I)`, we form the linear
interpolant

    .. math::  x_t = (1 - t)\\, x_0 + t\\, x_1, \\quad t \\sim \\mathcal{U}(0, 1)

and supervise a velocity network :math:`v_\\theta` to match the
analytic velocity :math:`v^\\star = x_1 - x_0`. Inference is a forward
ODE :math:`\\dot{x} = v_\\theta(x, t \\mid \\text{cond})` integrated from
:math:`t = 0` (noise) to :math:`t = 1` (data) with ``K`` Euler steps —
``K = 1`` corresponds to the requested *single-step* baseline.
"""

from __future__ import annotations

from typing import Tuple

import torch

from models import PVFlowMatcher


def fm_targets(x1: torch.Tensor, t: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the flow-matching interpolant and the target velocity.

    Args:
        x1: ``(B, N, C)`` ground-truth target (the HF–LF residual at points).
        t:  ``(B,)`` time scalars in ``[0, 1]``.

    Returns:
        Tuple ``(x_t, v_target)``:

        * ``x_t``      — ``(B, N, C)`` interpolated state.
        * ``v_target`` — ``(B, N, C)`` analytic velocity ``x1 - x0``.
    """
    x0 = torch.randn_like(x1)
    t_ = t.view(-1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    return x_t, (x1 - x0)


@torch.no_grad()
def direct_sample(model: PVFlowMatcher,
                  lf_voxel: torch.Tensor,
                  env: torch.Tensor,
                  style: torch.Tensor,
                  coords: torch.Tensor,
                  lf_pt: torch.Tensor) -> torch.Tensor:
    """One forward pass for the direct-regression variant.

    Used when cfg.flow.mode == "direct": the model is trained to predict
    the residual directly (no Euler integration). x_t is fed random noise
    (matching the training distribution) and t = 0.
    """
    B, N, _ = coords.shape
    t = torch.zeros(B, device=coords.device, dtype=coords.dtype)
    x_t = torch.randn(B, N, model.c_pt,
                      device=coords.device, dtype=coords.dtype)
    lf_feat, cond = model.encode_cond(lf_voxel, env, style, t)
    return model(x_t, coords, lf_pt, lf_feat, cond)


@torch.no_grad()
def lf_init_sample(model: PVFlowMatcher,
                   lf_voxel: torch.Tensor,
                   env: torch.Tensor,
                   style: torch.Tensor,
                   coords: torch.Tensor,
                   lf_pt: torch.Tensor,
                   steps: int = 1) -> torch.Tensor:
    """Inference for cfg.flow.mode == "lf_init".

    The model was trained with x_0 = LF disp, x_1 = HF, so the analytic
    velocity along the path is constant = HF - LF = residual. At
    inference we start at x = LF (no random noise) and integrate to
    estimate x_1 = HF. Returns the predicted RESIDUAL (HF - LF), matching
    the convention of the FM and direct samplers.
    """
    B, N, _ = coords.shape
    lf_disp_pt = lf_pt[..., :3]                         # (B, N, 3)
    x = lf_disp_pt.clone()
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=coords.device, dtype=coords.dtype)
        lf_feat, cond = model.encode_cond(lf_voxel, env, style, t)
        v = model(x, coords, lf_pt, lf_feat, cond)
        x = x + dt * v
    return x - lf_disp_pt                               # predicted residual


@torch.no_grad()
def euler_sample(model: PVFlowMatcher,
                 lf_voxel: torch.Tensor,
                 env: torch.Tensor,
                 style: torch.Tensor,
                 coords: torch.Tensor,
                 lf_pt: torch.Tensor,
                 steps: int = 1) -> torch.Tensor:
    """Integrate :math:`\\dot{x} = v_\\theta` with K Euler steps from noise → data.

    Args:
        model:    Trained :class:`models.PVFlowMatcher`.
        lf_voxel: ``(B, c_lf, D, D, D)`` LF crop.
        env:      ``(B, c_env, R, R, R)`` stitched LF env.
        style:    ``(B, n_style)`` cosmology vector.
        coords:   ``(B, N, 3)`` query point coords (any subset of the crop).
        lf_pt:    ``(B, N, c_lf_pt)`` LF disp at the points.
        steps:    Number of Euler steps. ``1`` = single-step.

    Returns:
        ``(B, N, c_pt)`` predicted target (HF–LF residual at the points).
    """
    B, N, _ = coords.shape
    x = torch.randn(B, N, model.c_pt,
                    device=coords.device, dtype=coords.dtype)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=coords.device, dtype=coords.dtype)
        lf_feat, cond = model.encode_cond(lf_voxel, env, style, t)
        v = model(x, coords, lf_pt, lf_feat, cond)
        x = x + dt * v
    return x
