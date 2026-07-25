#!/usr/bin/env python3
"""Field-level cosmology posterior and cross-fidelity KL (like the field CNN of the
CMASS section, adapted to our density fields).

For a field type (HF, LF, or SR) we train a small 3-D CNN moment network that
reads one density box and predicts a Gaussian posterior (mean and standard
deviation) for each cosmology parameter. We train one network on each field
type over the TRAIN boxes, then evaluate all of them on the held-out HR (HF)
TEST boxes, and report the per-parameter Gaussian KL of each network's posterior
against the HF-trained network's posterior. This asks: does a posterior model
built from cheap SR (or raw LF) fields transfer to real HF fields?

Densities come from ``eval_physical --dump_density`` (per-set npz with
rho_hf/rho_lf/rho_pred/style). Run after those dumps exist.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

PARAMS = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]


def load_fields(dirpath, key, res):
    """Load all boxes: (N, 1, res, res, res) arcsinh overdensity, and styles."""
    xs, ss = [], []
    for f in sorted(glob.glob(os.path.join(dirpath, "set*_density.npz"))):
        d = np.load(f)
        rho = d[key].astype(np.float32)
        # dumped field is ALREADY the overdensity (mean ~ 0); remove residual DC
        delta = rho - rho.mean()
        D = delta.shape[0]
        if D != res:                                  # average-pool to res
            f_ = D // res
            delta = delta.reshape(res, f_, res, f_, res, f_).mean((1, 3, 5))
        xs.append(np.arcsinh(delta))                  # tame the range, keep sign
        ss.append(d["style"].astype(np.float32))
    return np.stack(xs)[:, None], np.stack(ss)


class MomentCNN(nn.Module):
    """Small 3-D CNN -> (mu, log var) per parameter."""

    def __init__(self, n_out, base=16):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, 2, 1),
                                 nn.BatchNorm3d(co), nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(
            blk(1, base), blk(base, base * 2), blk(base * 2, base * 4),
            blk(base * 4, base * 4), nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.mu = nn.Linear(base * 4, n_out)
        self.logvar = nn.Linear(base * 4, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-10, 6)


def train_posterior(X, Y, ymean, ystd, device, epochs, val_frac=0.15, seed=0):
    """Train a moment network; return the model and its val NLL."""
    torch.manual_seed(seed)
    n = len(X); idx = torch.randperm(n)
    nv = max(4, int(n * val_frac))
    vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(X); Yt = torch.tensor((Y - ymean) / ystd)
    net = MomentCNN(Y.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (1e9, None)
    for ep in range(epochs):
        net.train()
        for b in torch.split(ti[torch.randperm(len(ti))], 16):
            xb = Xt[b].to(device); yb = Yt[b].to(device)
            mu, lv = net(xb)
            loss = (0.5 * ((yb - mu) ** 2) * torch.exp(-lv) + 0.5 * lv).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mu, lv = net(Xt[vi].to(device)); yb = Yt[vi].to(device)
            vnll = float((0.5 * ((yb - mu) ** 2) * torch.exp(-lv) + 0.5 * lv).mean())
        if vnll < best[0]:
            best = (vnll, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    return net, best[0]


@torch.no_grad()
def posterior(net, X, ymean, ystd, device):
    """Return (mu, sigma) in physical units for each box."""
    net.eval()
    mus, sigs = [], []
    for b in torch.split(torch.arange(len(X)), 16):
        mu, lv = net(torch.tensor(X[b]).to(device))
        mus.append(mu.cpu().numpy()); sigs.append(np.exp(0.5 * lv.cpu().numpy()))
    mu = np.concatenate(mus) * ystd + ymean
    sig = np.concatenate(sigs) * ystd
    return mu, sig


def gauss_kl(mu1, s1, mu2, s2):
    """KL( N(mu1,s1) || N(mu2,s2) ) per element."""
    return np.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out", default="figures_singlestep/field_kl.png")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training fields per type, and the HF test fields (everyone evaluated on HF).
    # Use "LF" (not "LR") for consistency with the rest of the section.
    keys = {"HF": "rho_hf", "SR": "rho_pred", "LF": "rho_lf"}
    Xtr = {n: load_fields(args.train_dir, k, args.res) for n, k in keys.items()}
    Xte_hf, Yte = load_fields(args.test_dir, "rho_hf", args.res)
    Ytr = Xtr["HF"][1]
    ymean, ystd = Ytr.mean(0), Ytr.std(0) + 1e-8
    print(f"[field-kl] train boxes {len(Ytr)}, test boxes {len(Yte)}, res {args.res}")

    # ensemble over seeds for stability
    post = {n: [] for n in keys}
    for n in keys:
        Xn, Yn = Xtr[n]
        for s in range(args.seeds):
            net, vnll = train_posterior(Xn, Yn, ymean, ystd, device, args.epochs, seed=s)
            post[n].append(posterior(net, Xte_hf, ymean, ystd, device))
            print(f"[field-kl] {n} seed {s}: val NLL {vnll:.3f}")

    def avg(n):
        mus = np.mean([p[0] for p in post[n]], 0)
        sigs = np.mean([p[1] for p in post[n]], 0)
        return mus, sigs
    muHF, sHF = avg("HF")
    # recovery correlation of the HF network (how informative the field is)
    for j, pn in enumerate(PARAMS):
        c = np.corrcoef(muHF[:, j], Yte[:, j])[0, 1]
        print(f"[field-kl] HF recovery corr {pn}: {c:+.3f}")

    print("\n[field-kl] cross-fidelity KL to the HF-trained posterior (lower better):")
    rows = {}
    for n in ["SR", "LF"]:
        mu, s = avg(n)
        kl = gauss_kl(muHF, sHF, mu, s).mean(0)   # per-param mean over test boxes
        rows[n] = kl
        print(f"  {n}: " + "  ".join(f"{pn}={k:.3f}" for pn, k in zip(PARAMS, kl))
              + f"   mean={kl.mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(PARAMS)); w = 0.35
    ax.bar(x - w / 2, rows["SR"], w, label="train on SR, test on HF", color="C2")
    ax.bar(x + w / 2, rows["LF"], w, label="train on LF, test on HF (control)", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(PARAMS); ax.set_ylabel("Gaussian KL to HF posterior")
    ax.set_title("Field-level cross-fidelity: does the posterior transfer to HF?")
    ax.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    np.savez(args.out.replace(".png", ".npz"), params=PARAMS,
             kl_sr=rows["SR"], kl_lf=rows["LF"])
    print(f"[field-kl] wrote {args.out}")


if __name__ == "__main__":
    main()
