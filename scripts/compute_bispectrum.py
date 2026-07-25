#!/usr/bin/env python3
"""Compute and plot the held-out bispectrum from dumped density fields.

Reads per-set density npz files written by ``eval_physical --dump_density``
(one directory per model, each holding rho_hf/rho_lf/rho_pred), computes the
equilateral and squeezed bispectrum of the overdensity for HF, LF, and each
model's SR field, averages over sets, and plots B(k) with the ratio to HF.

Usage:
    python scripts/compute_bispectrum.py \
        --model MSE:/path/mse --model GAN:/path/gan --model noz:/path/noz \
        --out figures_singlestep/bispectrum.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops.bispectrum import equilateral_bispectrum, squeezed_bispectrum  # noqa: E402


def _delta(rho):
    # The dumped field from disp_to_density is ALREADY the overdensity
    # (delta = rho/rho_bar - 1, mean ~ 0). Just remove any residual DC;
    # do NOT divide by the (near-zero) mean again.
    return rho - rho.mean()


def _mean_bispec(dirpath, key, fn, n_bins):
    ks, Bs = [], []
    for f in sorted(glob.glob(os.path.join(dirpath, "set*_density.npz"))):
        d = np.load(f)
        k, B = fn(_delta(d[key]), box_size=float(d["box_size"]), n_bins=n_bins)
        if len(k):
            ks.append(k); Bs.append(B)
    if not Bs:
        return None, None
    k = ks[0]
    B = np.stack([np.interp(k, kk, bb) for kk, bb in zip(ks, Bs)]).mean(0)
    return k, B


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="name:dir pairs; the first also supplies HF and LF")
    p.add_argument("--out", default="figures_singlestep/bispectrum.png")
    p.add_argument("--n_bins", type=int, default=12)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m.split(":", 1) for m in args.model]
    first_dir = models[0][1]
    cols = {"MSE": "C0", "GAN": "C3", "noz": "C2"}

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, (title, fn) in enumerate([
            ("Equilateral B(k,k,k)", equilateral_bispectrum),
            ("Squeezed B(k_soft,k,k)", squeezed_bispectrum)]):
        # HF and LF reference from the first model's dir
        kh, Bh = _mean_bispec(first_dir, "rho_hf", fn, args.n_bins)
        kl, Bl = _mean_bispec(first_dir, "rho_lf", fn, args.n_bins)
        ax[0, col].loglog(kh, np.abs(Bh), "k-", lw=2, label="HF (target)")
        ax[0, col].loglog(kl, np.abs(Bl), "--", color="gray", label="LF (input)")
        ax[1, col].axhline(1.0, color="k", lw=0.7, ls=":")
        ax[1, col].plot(kl, Bl / Bh, "--", color="gray", label="LF / HF")
        for name, d in models:
            k, B = _mean_bispec(d, "rho_pred", fn, args.n_bins)
            c = cols.get(name, None)
            ax[0, col].loglog(k, np.abs(B), color=c, label=f"SR ({name})")
            ax[1, col].plot(k, B / Bh, color=c, label=f"SR ({name}) / HF")
        ax[0, col].set_title(title + " (held out, never trained on)")
        ax[0, col].set_ylabel("|B(k)|"); ax[0, col].legend(fontsize=7)
        ax[1, col].set_xlabel("k [h/Mpc]"); ax[1, col].set_ylabel("ratio to HF")
        ax[1, col].set_ylim(0, 2); ax[1, col].legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
