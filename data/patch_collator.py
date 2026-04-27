"""Custom collator that stacks the dict-of-tensors samples into a batch."""

from __future__ import annotations

from typing import Sequence

import torch


_TENSOR_KEYS = (
    "lf_voxel", "env", "coords", "lf_pt", "tgt_pt",
    "tgt_vox", "loss_mask", "pt_mask", "style",
)
_META_KEYS = ("set_id", "crop_xyz", "extent")


class PatchCollator:
    """Callable that batches a list of :class:`SimulationDataset` samples.

    Stacks every tensor field along ``dim=0`` and gathers metadata
    (``set_id``, ``crop_xyz``, ``extent``) into Python lists. Use as
    ``DataLoader(..., collate_fn=PatchCollator())``.
    """

    def __call__(self, batch: Sequence[dict]) -> dict:
        out: dict = {}
        for k in _TENSOR_KEYS:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        for k in _META_KEYS:
            out[k] = [b[k] for b in batch]
        return out
