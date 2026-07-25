#!/usr/bin/env python3
"""Apply the cosmology-conditioned amplitude correction to dumped SR densities.

Reads per-set density npz (rho_hf/rho_lf/rho_pred/style/box_size), predicts the
transfer curve T(k) from the box's cosmology via the fitted emulator, and
rescales the SR overdensity's per-shell Fourier amplitude by 1/T(k)
(phase-preserving). Writes the same npz with rho_pred replaced by the corrected
field, so the downstream posterior can train on the DEPLOYMENT (corrected) SR.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops.spectrum import rescale_by_curve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--emulator", required=True)
    args = p.parse_args()
    ez = np.load(args.emulator)
    kc, coef = ez["k"], ez["coef"]        # coef: (n_bins, 1+5)
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(args.in_dir, "set*_density.npz"))):
        d = np.load(f)
        style = d["style"].astype(np.float64)
        T_hat = coef @ np.concatenate([[1.0], style])   # emulator T(k) from cosmology
        delta = d["rho_pred"] - d["rho_pred"].mean()     # already overdensity
        corr = rescale_by_curve(delta, kc, T_hat, box_size=float(d["box_size"]))
        np.savez_compressed(os.path.join(args.out_dir, os.path.basename(f)),
                            rho_hf=d["rho_hf"], rho_lf=d["rho_lf"],
                            rho_pred=corr.astype(np.float32),
                            box_size=float(d["box_size"]), L=int(d["L"]),
                            style=d["style"])
        n += 1
    print(f"[correct] wrote {n} corrected-SR density files -> {args.out_dir}")


if __name__ == "__main__":
    main()
