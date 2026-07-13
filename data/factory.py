"""High-level factory: build datasets + dataloaders from a Config."""

from __future__ import annotations

import os
from typing import Optional

from torch.utils.data import DataLoader, RandomSampler

from config import Config

from .normalization import (NormStats, compute_norm_stats,
                            compute_residual_stats)
from .patch_collator import PatchCollator
from .readers import TileReader, get_reader
from .simulation_dataset import (SNAPSHOT_DEFAULT, SimulationDataset,
                                 discover_sets, split_sets)

# How many tiles to sample when estimating normalization stats
# (override via cfg["data"]["norm_sample_tiles"]). One tile per set,
# sets strided evenly across the (sorted) train split so the sample
# spans the cosmology range (per-set displacement std varies ~4x with
# sigma_8 — stats from a single set would be badly biased).
_NORM_SAMPLE_TILES = 32


def _sample_sets(sets, k: int):
    """Evenly-strided subsample of ``k`` sets from the (ordered) list."""
    if len(sets) <= k:
        return list(sets)
    stride = len(sets) / k
    return [sets[int(i * stride)] for i in range(k)]


def _tile_path(root: str, sub: str, sid: int, snapshot: str,
               field: str, tile=(0, 0, 0)) -> str:
    ix, iy, iz = tile
    return os.path.join(root, sub, f"set{sid}_pos_{ix}_{iy}_{iz}",
                        snapshot, f"{field}.npy")


def _lf_tile_paths(root: str, sets, snapshot: str, field: str = "disp",
                   k: int = _NORM_SAMPLE_TILES) -> list[str]:
    """One LF tile per set for up to ``k`` sets, strided across the split."""
    return [_tile_path(root, "quijotelike-64", sid, snapshot, field)
            for sid, _ in _sample_sets(sets, k)]


def _paired_tile_paths(root: str, sets, snapshot: str,
                       k: int = _NORM_SAMPLE_TILES
                       ) -> list[tuple[str, str]]:
    """(LF, HF) disp tile pairs, one per set, for residual stats."""
    return [(_tile_path(root, "quijotelike-64", sid, snapshot, "disp"),
             _tile_path(root, "quijote-64", sid, snapshot, "disp"))
            for sid, _ in _sample_sets(sets, k)]


def _compute_full_norm(root: str, train_sets, snapshot: str,
                       k: int = _NORM_SAMPLE_TILES) -> NormStats:
    """Input stats from LF train tiles + residual stats from LF/HF pairs."""
    norm = compute_norm_stats(
        _lf_tile_paths(root, train_sets, snapshot, k=k), max_files=k)
    res_mean, res_std = compute_residual_stats(
        _paired_tile_paths(root, train_sets, snapshot, k=k), max_files=k)
    norm.res_mean = res_mean
    norm.res_std = res_std
    return norm


def build_norm_stats(cfg: Config, reader: TileReader) -> NormStats:
    """Compute norm stats from the *training* split of the configured root."""
    snapshot = cfg["data"].get("snapshot", SNAPSHOT_DEFAULT)
    sets = discover_sets(cfg["data"]["root"], reader, snapshot)
    splits = split_sets(sets)
    k = int(cfg["data"].get("norm_sample_tiles", _NORM_SAMPLE_TILES))
    return _compute_full_norm(cfg["data"]["root"], splits["train"], snapshot, k)


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
        k = int(cfg["data"].get("norm_sample_tiles", _NORM_SAMPLE_TILES))
        norm = _compute_full_norm(cfg["data"]["root"], splits["train"],
                                  snapshot, k)

    fields = list(cfg["data"].get("fields", ["disp"]))
    if fields[0] != "disp":
        raise ValueError(f"cfg.data.fields[0] must be 'disp'; got {fields}")
    if extra_norms is None:
        extra_norms = {}
        for f in fields[1:]:
            paths = _lf_tile_paths(cfg["data"]["root"], splits["train"],
                                   snapshot, field=f)
            extra_norms[f] = compute_norm_stats(
                paths, max_files=_NORM_SAMPLE_TILES)

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

    # persistent_workers keeps the per-worker env/style caches warm across
    # epochs and avoids the per-epoch worker spawn cost. prefetch_factor>2
    # hides per-sample I/O latency on networked storage.
    prefetch = cfg["train"].get("prefetch_factor", 4)
    # Optional fast-prototype knob: subsample the train epoch to N random
    # crops (with replacement) instead of iterating all ~224k. Useful for
    # quickly validating an arch change without paying for a full epoch
    # on contended storage.
    max_train = int(cfg["train"].get("max_train_crops", 0) or 0)
    loaders: dict[str, DataLoader] = {}
    for name, ds in datasets.items():
        kw = dict(
            batch_size=bs,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=collate,
            drop_last=(name == "train"),
        )
        if name == "train" and max_train > 0 and max_train < len(ds):
            # RandomSampler with replacement + num_samples caps the epoch
            kw["sampler"] = RandomSampler(ds, replacement=True,
                                          num_samples=max_train)
        else:
            kw["shuffle"] = (name == "train")
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = prefetch
        loaders[name] = DataLoader(ds, **kw)
    return loaders
