#!/usr/bin/env python3
"""Fit a cosmology-conditioned transfer-function emulator T(k) = f(style).

Per k-bin least-squares on features [1, style(5)] from a per-set dump
(eval_physical --dump_tk_perset). Saves (k, coef) for --correct_emulator.
"""
import sys
import numpy as np

d = np.load(sys.argv[1])
style, T, k = d["style"], d["T"], d["k"]           # (N,5), (N,nb), (nb,)
F = np.concatenate([np.ones((len(style), 1)), style], axis=1)   # (N,6)
coef = np.zeros((T.shape[1], F.shape[1]))
for b in range(T.shape[1]):
    coef[b] = np.linalg.lstsq(F, T[:, b], rcond=None)[0]
np.savez(sys.argv[2], k=k, coef=coef)
# leave-one-out sanity: mean |T_pred - T_true| over bins
resid = []
for i in range(len(style)):
    m = np.ones(len(style), bool); m[i] = False
    ci = np.stack([np.linalg.lstsq(F[m], T[m, b], rcond=None)[0]
                   for b in range(T.shape[1])])
    resid.append(np.mean(np.abs(ci @ F[i] - T[i])))
print(f"[fit] {len(style)} sims, {T.shape[1]} bins; LOO mean |dT|={np.mean(resid):.4f}")
