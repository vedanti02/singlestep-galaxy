"""Plot train/val loss trajectories for all runs, plus rel_to_zero per epoch.

Usage:
    python plot_runs.py
    python plot_runs.py --runs_dir runs --pattern 'pvfm_*' --out plots/all.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_log(run_dir: Path):
    p = run_dir / "log.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p) as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return log


def _read_per_epoch_rel(run_dir: Path) -> dict:
    """Return {epoch_int: rel_to_zero (inner-cube aggregate)} for diagnostics found."""
    out = {}
    for csv_path in run_dir.glob("**/boundary_error.csv"):
        epoch_dir = csv_path.parent.name
        if not epoch_dir.startswith("epoch"):
            continue
        try:
            ep = int(epoch_dir.replace("epoch", ""))
        except ValueError:
            continue
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        total_err = total_zero = total_cnt = 0.0
        for r in rows:
            d = int(r["distance_from_edge"])
            if d < 4:
                continue
            n = float(r["count"])
            total_err += float(r["mean_sq_error"]) * n
            total_zero += float(r["baseline_zero"]) * n
            total_cnt += n
        if total_cnt > 0 and total_zero > 0:
            out[ep] = (total_err / total_cnt) / (total_zero / total_cnt)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--pattern", type=str, default="pvfm_*")
    p.add_argument("--out", type=str, default="plots/all_runs.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_train, ax_val, ax_pt, ax_rel = axes.ravel()

    runs = []
    for d in sorted(Path(args.runs_dir).glob(args.pattern)):
        if not d.is_dir():
            continue
        log = _read_log(d)
        if log is None:
            continue
        runs.append((d.name, d, log))

    cmap = plt.get_cmap("tab20")
    for i, (name, d, log) in enumerate(runs):
        color = cmap(i % 20)
        train = log.get("train", [])
        val = log.get("val", [])
        if train:
            ax_train.plot([t["epoch"] for t in train],
                          [t["loss"] for t in train],
                          label=name, color=color, lw=1.4)
            ax_pt.plot([t["epoch"] for t in train],
                       [t["pt_loss"] for t in train],
                       color=color, lw=1.0, ls="--", alpha=0.7)
        if val:
            ax_val.plot([v.get("epoch", j) for j, v in enumerate(val)],
                        [v["loss"] for v in val],
                        label=name, color=color, lw=1.4, marker="o", ms=3)
            ax_pt.plot([v.get("epoch", j) for j, v in enumerate(val)],
                       [v["pt_loss"] for v in val],
                       color=color, lw=1.4, marker="o", ms=3, label=name)
        rel = _read_per_epoch_rel(d)
        if rel:
            xs = sorted(rel.keys())
            ax_rel.plot(xs, [rel[x] for x in xs],
                        label=name, color=color, lw=1.4, marker="s", ms=4)

    for ax, title, ylabel in [
            (ax_train, "Train total loss", "loss"),
            (ax_val,   "Val total loss",   "loss"),
            (ax_pt,    "pt_loss (val solid, train dashed)", "pt_loss"),
            (ax_rel,   "Boundary diagnostic rel_to_zero", "rel_to_zero"),
            ]:
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
    ax_rel.axhline(1.0, color="k", lw=0.8, ls=":", label="baseline=1")
    ax_pt.axhline(0.73, color="k", lw=0.8, ls=":", label="direct baseline=0.73")
    ax_pt.axhline(1.7,  color="k", lw=0.8, ls="--", label="FM baseline=1.7")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[plot_runs] wrote {out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
