"""Plot rel_to_zero vs distance-from-boundary for multiple runs.

Surfaces the headline diagnostic: which models beat predict-zero (rel<1)
at which crop locations.

Usage:
    python plot_boundary.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(p: Path):
    try:
        with open(p) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    return rows


def _newest_boundary(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("**/boundary_error.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--pattern", type=str, default="pvfm_*")
    p.add_argument("--out", type=str, default="plots/boundary.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_abs, ax_rel = axes

    cmap = plt.get_cmap("tab10")
    n = 0
    for d in sorted(Path(args.runs_dir).glob(args.pattern)):
        if not d.is_dir():
            continue
        latest = _newest_boundary(d)
        if latest is None:
            continue
        rows = _read_csv(latest)
        if not rows:
            continue
        dists = np.array([int(r["distance_from_edge"]) for r in rows])
        model = np.array([float(r["mean_sq_error"]) for r in rows])
        zero  = np.array([float(r["baseline_zero"]) for r in rows])
        rel   = np.array([float(r["rel_to_zero"]) for r in rows])

        color = cmap(n % 10)
        n += 1
        ax_abs.plot(dists, model, color=color, lw=1.5,
                    label=f"{d.name} model")
        ax_abs.plot(dists, zero, color=color, lw=0.8, ls="--",
                    alpha=0.5)
        ax_rel.plot(dists, rel, color=color, lw=1.5,
                    marker="o", ms=4, label=d.name)

    ax_abs.set_title("Per-distance MSE")
    ax_abs.set_xlabel("L_inf distance from crop boundary")
    ax_abs.set_ylabel("mean squared error")
    ax_abs.legend(fontsize=7, loc="best")
    ax_abs.grid(alpha=0.3)
    ax_abs.axvline(4, color="k", lw=0.8, ls=":", alpha=0.5)  # buffer boundary
    ax_abs.text(4.1, ax_abs.get_ylim()[1] * 0.9, "buffer | inner", fontsize=8)

    ax_rel.set_title("rel_to_zero by distance (<1 = beats predict-zero)")
    ax_rel.set_xlabel("L_inf distance from crop boundary")
    ax_rel.set_ylabel("model_mse / zero_mse")
    ax_rel.axhline(1.0, color="k", lw=1.2, ls=":", label="baseline=1")
    ax_rel.axvline(4, color="k", lw=0.8, ls=":", alpha=0.5)
    ax_rel.legend(fontsize=7, loc="best")
    ax_rel.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[plot_boundary] wrote {out}  ({n} runs)")


if __name__ == "__main__":
    main()
