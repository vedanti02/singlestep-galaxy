"""High-level factory: build datasets + dataloaders from a Config."""

from __future__ import annotations

import os
from typing import Optional

from torch.utils.data import DataLoader

from config import Config

from .normalization import NormStats, compute_norm_stats
from .patch_collator import PatchCollator
from .readers import TileReader, get_reader
from .simulation_dataset import (SNAPSHOT_DEFAULT, SimulationDataset,
                                 discover_sets, split_sets)


def _stitched_lf_paths(root: str, sets, snapshot: str) -> list[str]:
    return [os.path.join(root, "stitched", f"set{sid}_quijotelike",
                         snapshot, "disp.npy")
            for sid, _ in sets]


def _quijotelike_field_paths(root: str, sets, snapshot: str,
                             field: str) -> list[str]:
    """Per-set tile paths for an arbitrary field (e.g. 'vel').

    Falls back to the (1,1,1) tile when only one is present, otherwise
    enumerates all tiles for the set's extent so norm stats are
    computed on a representative spatial sample.
    """
    paths: list[str] = []
    for sid, ext in sets:
        ex, ey, ez = ext
        for ix in range(ex):
            for iy in range(ey):
                for iz in range(ez):
                    paths.append(os.path.join(
                        root, "quijotelike-64",
                        f"set{sid}_pos_{ix}_{iy}_{iz}",
                        snapshot, f"{field}.npy"
                    ))
    return paths


def build_norm_stats(cfg: Config, reader: TileReader) -> NormStats:
    """Compute norm stats from the *training* split of the configured root."""
    snapshot = cfg["data"].get("snapshot", SNAPSHOT_DEFAULT)
    sets = discover_sets(cfg["data"]["root"], reader, snapshot)
    splits = split_sets(sets)
    paths = _stitched_lf_paths(cfg["data"]["root"], splits["train"], snapshot)
    return compute_norm_stats(paths, max_files=16)


def build_datasets(cfg: Config,
                   reader: Optional[TileReader] = None,
                   norm: Optional[NormStats] = None,
                   extra_norms: Optional[dict] = None,
                   ) -> tuple[dict[str, SimulationDataset], NormStats, dict]:
    """Build {'train','val','test'} datasets from a Config.

    Returns:
        Tuple ``(datasets, norm, extra_norms)``. ``datasets`` is a dict
        keyed by split name; missing splits map to empty dicts.
        ``extra_norms`` is a dict ``{field: NormStats}`` for any extra
        LF input fields beyond "disp" (empty if cfg.data.fields == ["disp"]).
    """
    reader = reader or get_reader(cfg["data"].get("reader", "numpy"))
    snapshot = cfg["data"].get("snapshot", SNAPSHOT_DEFAULT)
    sets = discover_sets(cfg["data"]["root"], reader, snapshot)
    splits = split_sets(sets)

    if norm is None:
        train_paths = _stitched_lf_paths(cfg["data"]["root"], splits["train"], snapshot)
        norm = compute_norm_stats(train_paths, max_files=16)

    fields = list(cfg["data"].get("fields", ["disp"]))
    if fields[0] != "disp":
        raise ValueError(f"cfg.data.fields[0] must be 'disp'; got {fields}")
    if extra_norms is None:
        extra_norms = {}
        for f in fields[1:]:
            paths = _quijotelike_field_paths(
                cfg["data"]["root"], splits["train"], snapshot, f)
            extra_norms[f] = compute_norm_stats(paths, max_files=16)

    aug = bool(cfg["data"].get("augment", False))

    out: dict[str, SimulationDataset] = {}
    for name, set_list in splits.items():
        if not set_list:
            continue
        ds = SimulationDataset(
            root=cfg["data"]["root"],
            sets=set_list,
            crop_size=cfg["data"]["crop_size"],
            crop_overlap=cfg["data"]["crop_overlap"],
            norm_stats=norm,
            env_outside_mask=cfg["data"].get("env_outside_mask", True),
            env_resolution=cfg["model"].get("env_resolution", 64),
            reader=reader,
            snapshot=snapshot,
            seed=cfg["train"].get("seed", 0) + (0 if name == "train" else 1),
            fields=fields,
            extra_norms=extra_norms,
        )
        # Only augment the train split — val/test must stay deterministic.
        ds.augment = aug and (name == "train")
        out[name] = ds
    return out, norm, extra_norms


def build_dataloaders(cfg: Config,
                      datasets: dict[str, SimulationDataset]
                      ) -> dict[str, DataLoader]:
    """Wrap each dataset in a DataLoader using ``cfg['train']`` settings."""
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 0)
    pin = cfg["train"].get("device", "cpu").startswith("cuda")
    collate = PatchCollator()

    loaders: dict[str, DataLoader] = {}
    for name, ds in datasets.items():
        loaders[name] = DataLoader(
            ds, batch_size=bs,
            shuffle=(name == "train"),
            num_workers=nw,
            pin_memory=pin,
            collate_fn=collate,
            drop_last=(name == "train"),
        )
    return loaders
