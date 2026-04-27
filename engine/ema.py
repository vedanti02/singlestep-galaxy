"""Plain exponential-moving-average shadow of model parameters."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of an ``nn.Module``'s parameters.

    Maintains a frozen ``shadow`` model whose parameters are
    :math:`\\theta_{\\text{ema}} \\leftarrow \\alpha\\, \\theta_{\\text{ema}}
    + (1 - \\alpha)\\, \\theta`. Use :meth:`shadow_state_dict` for
    eval / checkpointing.

    Args:
        model: Live model to track.
        decay: EMA decay :math:`\\alpha` in :math:`[0, 1)`. Higher → smoother.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd_live = model.state_dict()
        sd_ema  = self.shadow.state_dict()
        for k, v in sd_live.items():
            if v.dtype.is_floating_point:
                sd_ema[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:                                                # int buffers etc.
                sd_ema[k].copy_(v)

    def shadow_state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_shadow(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)
