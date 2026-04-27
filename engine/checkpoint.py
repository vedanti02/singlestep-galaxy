"""Tiny checkpoint manager: writes ``ckpt_latest.pt`` + epoch snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from data.normalization import NormStats


class CheckpointManager:
    """Save / restore ``(model, ema, optim, epoch, norm, cfg)`` bundles."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, *, epoch: int,
             model: nn.Module,
             optim: torch.optim.Optimizer,
             norm: NormStats,
             cfg: dict,
             ema_state: Optional[dict] = None,
             tag: Optional[str] = None) -> Path:
        """Write a checkpoint and refresh ``ckpt_latest.pt``.

        Args:
            tag: If given, also writes ``ckpt_{tag}.pt`` (e.g. "epoch042").
        """
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema":   ema_state,
            "optim": optim.state_dict(),
            "norm":  norm.to_dict(),
            "cfg":   cfg,
        }
        latest = self.out_dir / "ckpt_latest.pt"
        torch.save(payload, latest)
        if tag is not None:
            torch.save(payload, self.out_dir / f"ckpt_{tag}.pt")
        return latest

    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)
