"""Loss functions for the supervised single-step PVD baseline.

Why MSE-on-residual?
--------------------
The supervisory signal here is the **per-cell HF–LF displacement
residual** in normalized units. Two physical facts drive the choice of
plain MSE:

1. *The residual is approximately Gaussian and zero-mean.* After
   subtracting the LF displacement and rescaling by per-channel
   :math:`\\sigma`, the per-voxel residual distribution is unimodal,
   centred at 0, with light tails. Maximum-likelihood estimation under
   a Gaussian noise model is exactly MSE — no missing constant, no
   reweighting needed. (A Poisson NLL would be appropriate if the
   target were a *count* field — e.g., halo occupancy on a grid — but
   here the target lives in :math:`\\mathbb{R}^3` and can take either
   sign.)

2. *Permutation does not matter.* The model predicts at the same
   Lagrangian indices as the target — point :math:`i` in the prediction
   *is* point :math:`i` in the target. There is no point-set matching
   ambiguity, so a Chamfer / EMD loss (whose role is to handle unordered
   sets) would only add noise. Chamfer would matter if we were
   predicting *Eulerian* particle positions without an index
   correspondence.

The two loss terms below implement this:

* :func:`masked_pt_mse` — primary point-wise MSE on the velocity
  prediction, with the per-point mask zeroing out points that lie in
  the ``d/2`` edge buffer.
* :func:`voxel_consistency_mse` — auxiliary MSE that scatters the
  point-wise estimate of :math:`x_1` back onto the voxel grid and
  matches the full HF–LF residual cube. This regularises the model to
  produce voxel-consistent predictions when many points share a cell.
"""

from __future__ import annotations

import torch

from ops.geometry import points_to_voxel


def masked_pt_mse(v_pred: torch.Tensor, v_target: torch.Tensor,
                  pt_mask: torch.Tensor) -> torch.Tensor:
    """Per-point MSE with edge-buffer masking.

    Args:
        v_pred:   ``(B, N, C)`` predicted flow velocity at points.
        v_target: ``(B, N, C)`` analytic FM target ``x_1 - x_0``.
        pt_mask:  ``(B, N)`` 0/1 mask — 1 where the point lies inside
            the inner ``D - d`` cube (so the loss ignores edge points).

    Returns:
        Scalar tensor: mean squared error over masked entries.
    """
    m = pt_mask.unsqueeze(-1)                                    # (B, N, 1)
    diff2 = (v_pred - v_target) ** 2 * m
    den = m.expand_as(diff2).sum().clamp(min=1.0)
    return diff2.sum() / den


def divergence_mse(disp_pred_vox: torch.Tensor,
                   disp_true_vox: torch.Tensor,
                   loss_mask: torch.Tensor) -> torch.Tensor:
    """MSE on the divergence of a 3-component displacement field.

    To first order in displacement, the density contrast is
    :math:`\\delta = -\\nabla \\cdot \\boldsymbol{u}`. Matching the
    divergence is therefore matching the linearized density. We use
    central differences with periodic boundary conditions (torch.roll),
    apply the inner-cube mask, and return mean squared error.

    Args:
        disp_pred_vox: ``(B, 3, D, D, D)`` predicted displacement field.
        disp_true_vox: ``(B, 3, D, D, D)`` ground-truth displacement.
        loss_mask:     ``(B, D, D, D)`` 0/1 inner-cube mask.

    Returns:
        Scalar tensor.
    """
    def _div(u):
        ddx = (torch.roll(u[:, 0], -1, dims=1)
               - torch.roll(u[:, 0],  1, dims=1)) * 0.5
        ddy = (torch.roll(u[:, 1], -1, dims=2)
               - torch.roll(u[:, 1],  1, dims=2)) * 0.5
        ddz = (torch.roll(u[:, 2], -1, dims=3)
               - torch.roll(u[:, 2],  1, dims=3)) * 0.5
        return ddx + ddy + ddz                                       # (B, D, D, D)

    div_pred = _div(disp_pred_vox)
    div_true = _div(disp_true_vox)
    diff2 = (div_pred - div_true) ** 2 * loss_mask
    den = loss_mask.sum().clamp(min=1.0)
    return diff2.sum() / den


def voxel_consistency_mse(x1_pt_hat: torch.Tensor,
                          coords: torch.Tensor,
                          tgt_vox: torch.Tensor,
                          loss_mask: torch.Tensor) -> torch.Tensor:
    """Auxiliary voxel-grid MSE on the implied :math:`x_1` estimate.

    From the FM interpolant :math:`x_t = (1-t) x_0 + t x_1` and the
    predicted velocity :math:`v = x_1 - x_0` we can read off
    :math:`\\hat{x}_1 = x_t + (1 - t)\\, v`. Scatter that estimate onto
    a voxel grid (averaging within each cell — see
    :func:`ops.geometry.points_to_voxel`) and compare against the
    ground-truth HF–LF residual cube on cells inside the inner cube.

    Args:
        x1_pt_hat: ``(B, N, C)`` per-point estimate of the residual.
        coords:    ``(B, N, 3)`` point coordinates in ``[0, 1]``.
        tgt_vox:   ``(B, C, D, D, D)`` ground-truth normalized residual.
        loss_mask: ``(B, D, D, D)`` 0/1 voxel mask (inner cube only).

    Returns:
        Scalar tensor.
    """
    D = tgt_vox.shape[-1]
    x1_vox_hat = points_to_voxel(coords, x1_pt_hat, R=D, reduction="mean")
    vm = loss_mask.unsqueeze(1)                                  # (B, 1, D, D, D)
    diff2 = (x1_vox_hat - tgt_vox) ** 2 * vm
    den = vm.expand_as(diff2).sum().clamp(min=1.0)
    return diff2.sum() / den


def gradient_penalty(critic, real_in: torch.Tensor, fake_in: torch.Tensor,
                     style: torch.Tensor | None = None) -> torch.Tensor:
    """WGAN-GP gradient penalty on interpolates of the full critic input.

    Interpolates the whole concatenated tensor (field + conditioning
    channels). For paired conditional critics the conditioning channels
    are identical between ``real_in`` and ``fake_in``, so they interpolate
    to themselves and the penalty constrains the critic's Lipschitz
    constant jointly over field and conditioning — the standard cGAN-GP
    treatment.

    Must run OUTSIDE autocast and without a GradScaler: the
    ``create_graph=True`` double-backward and scaled gradients do not mix
    (the trainer keeps all critic math in fp32 for this reason).

    Args:
        critic:  Callable ``(B, C, S, S, S) [, style] -> (B,)``.
        real_in: ``(B, C, S, S, S)`` real critic input.
        fake_in: ``(B, C, S, S, S)`` fake critic input (detached).
        style:   Optional ``(B, style_dim)`` passed through to the critic.

    Returns:
        Scalar ``E[(||grad|| - 1)^2]``.
    """
    B = real_in.shape[0]
    eps = torch.rand(B, 1, 1, 1, 1, device=real_in.device,
                     dtype=real_in.dtype)
    x = (eps * real_in + (1.0 - eps) * fake_in).requires_grad_(True)
    score = critic(x, style)
    (grad,) = torch.autograd.grad(score.sum(), x, create_graph=True)
    return ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
