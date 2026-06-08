"""Generate paper-focused figures from existing CSVs / log.json.

Produces, under plots/paper/:

  fig_boundary_compare.png  — rel_to_zero(d) for 3 ckpts: lfinit (old best),
                              tbias (degenerate), noskip ep9 (new best).
  fig_boundary_traj.png     — inner_rel vs epoch for the full noskip run
                              (Fig 4 in the paper §5.9.3).
  fig_spectra.png           — T(k) and r(k) curves per val set for noskip ep9,
                              vs the LF baseline (Fig in §5.9.1).
  fig_loss_traj.png         — train/val pt-MSE vs epoch for noskip run only
                              (capacity ceiling plot for §5.9.2).

All plots use a uniform colour scheme: navy = model, grey = LF baseline,
gold = old-best lfinit, red = tbias (degenerate).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# ---------- styling ----------
MODEL  = "#1f3a93"     # navy
LF     = "#888888"     # grey
OLD    = "#d4a017"     # gold
BAD    = "#c0392b"     # red
HF     = "#27ae60"     # green
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})


def read_boundary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    dist = np.array([int(r["distance_from_edge"]) for r in rows])
    mse  = np.array([float(r["mean_sq_error"]) for r in rows])
    zer  = np.array([float(r["baseline_zero"]) for r in rows])
    cnt  = np.array([int(r["count"]) for r in rows])
    return dist, mse, zer, cnt


def inner_rel(path: Path, buf: int = 4) -> float:
    dist, mse, zer, cnt = read_boundary(path)
    keep = dist >= buf
    return float((mse[keep] * cnt[keep]).sum() / (zer[keep] * cnt[keep]).sum())


# ---------- Fig: boundary comparison ----------
def fig_boundary_compare(out: Path):
    triples = [
        ("lfinit (12 ep)",      "runs/pvfm_lfinit_7878905/boundary/boundary_error.csv", OLD),
        ("tbias  (12 ep)",      "runs/pvfm_tbias_7889491/boundary/boundary_error.csv",  BAD),
        ("noskip ep9 (best)",   "runs/pvfm_noskip_8137633/boundary/epoch009/boundary_error.csv", MODEL),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for label, p, col in triples:
        if not Path(p).exists():
            print(f"  [skip] {p}")
            continue
        dist, mse, zer, cnt = read_boundary(Path(p))
        rel = mse / zer
        ax.plot(dist, rel, marker="o", ms=4, color=col, label=label)
    ax.axhline(1.0, ls="--", lw=1, c="k", alpha=0.5, label="predict-zero")
    ax.axvspan(0, 3.5, alpha=0.08, color="red")  # buffer region
    ax.text(1.5, ax.get_ylim()[1]*0.93, "buffer\n(d/2)",
            ha="center", fontsize=8, color="darkred")
    ax.set_xlabel(r"L$_\infty$ distance to region face (voxels)")
    ax.set_ylabel(r"rel_to_zero = MSE / $\|$residual$\|^2$")
    ax.set_title("Boundary diagnostic: model vs prior baselines")
    ax.set_xticks(range(0, 16, 2))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- Fig: boundary trajectory ----------
def fig_boundary_traj(out: Path):
    base = Path("runs/pvfm_noskip_8137633/boundary")
    eps, rels = [], []
    for ep in range(30):
        p = base / f"epoch{ep:03d}" / "boundary_error.csv"
        if p.exists():
            eps.append(ep); rels.append(inner_rel(p))
    # Also overlay the prior baselines as horizontal lines
    base_lfinit = inner_rel(Path("runs/pvfm_lfinit_7878905/boundary/boundary_error.csv"))
    base_tbias  = inner_rel(Path("runs/pvfm_tbias_7889491/boundary/boundary_error.csv"))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(eps, rels, marker="o", color=MODEL, label="noskip (this work)")
    best_e = int(np.argmin(rels)); best_r = rels[best_e]
    ax.scatter([eps[best_e]], [best_r], s=80, edgecolor="k", facecolor="gold",
               zorder=5, label=f"best: ep{eps[best_e]} = {best_r:.3f}")
    ax.axhline(base_lfinit, color=OLD, ls="--", lw=1.2,
               label=f"lfinit best = {base_lfinit:.3f}")
    ax.axhline(base_tbias, color=BAD, ls="--", lw=1.2,
               label=f"tbias = {base_tbias:.3f}")
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="predict-zero")
    ax.set_xlabel("epoch")
    ax.set_ylabel("inner-cube rel_to_zero")
    ax.set_title("Convergence of the architectural-fix run (20 epochs)")
    ax.set_xticks(range(0, 21, 2))
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- Fig: spectra T(k), r(k) ----------
def fig_spectra(out: Path):
    run_dir = Path("runs/pvfm_noskip_8137633/physical")
    sets = [18, 28, 38]
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), sharex=True)
    for col, sid in enumerate(sets):
        p = run_dir / f"set{sid}_spectra.csv"
        if not p.exists():
            for r in (0, 1):
                axes[r, col].text(0.5, 0.5, f"set{sid}: no data",
                                  ha="center", transform=axes[r, col].transAxes)
                axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            continue
        rows = list(csv.DictReader(open(p)))
        k     = np.array([float(r["k"])      for r in rows])
        T_lf  = np.array([float(r["T_lf"])   for r in rows])
        T_pr  = np.array([float(r["T_pred"]) for r in rows])
        r_lf  = np.array([float(r["r_lf"])   for r in rows])
        r_pr  = np.array([float(r["r_pred"]) for r in rows])

        # top row: T(k)
        ax = axes[0, col]
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.plot(k, T_lf, color=LF,    label="LF baseline")
        ax.plot(k, T_pr, color=MODEL, label="model (ep9)")
        ax.set_xscale("log")
        ax.set_title(f"set {sid}")
        if col == 0: ax.set_ylabel(r"$T(k)$  (amplitude)")
        ax.set_ylim(0.6, 1.6)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.95)

        # bottom row: r(k)
        ax = axes[1, col]
        ax.plot(k, r_lf, color=LF,    label="LF baseline")
        ax.plot(k, r_pr, color=MODEL, label="model (ep9)")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
        if col == 0: ax.set_ylabel(r"$r(k)$  (coherence)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.95)

    fig.suptitle("Spectral diagnostics: transfer function and cross-coherence (best ckpt)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- Fig: loss trajectory ----------
def fig_loss_traj(out: Path):
    log = json.load(open("runs/pvfm_noskip_8137633/log.json"))
    tr = log["train"]; va = log["val"]
    ep_t = [r["epoch"] for r in tr]; pt_t = [r["pt_loss"] for r in tr]
    ep_v = [r["epoch"] for r in va]; pt_v = [r["pt_loss"] for r in va]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(ep_t, pt_t, marker="o", color=MODEL, label="train pt-MSE")
    ax.plot(ep_v, pt_v, marker="s", color=OLD,   label="val pt-MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("pt-MSE")
    ax.set_xticks(range(0, 21, 2))
    ax.set_title("Train / val per-particle MSE (capacity-ceiling plot)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    out_dir = Path("plots/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_boundary_compare(out_dir / "fig_boundary_compare.png")
    fig_boundary_traj(   out_dir / "fig_boundary_traj.png")
    fig_spectra(         out_dir / "fig_spectra.png")
    fig_loss_traj(       out_dir / "fig_loss_traj.png")
