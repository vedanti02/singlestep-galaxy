"""Evaluation CLI entry point.

Usage:
    python eval.py --ckpt runs/pvfm/ckpt_latest.pt
    python eval.py --ckpt ... --max_sets 3 --steps 4 --use_ema
    python eval.py --ckpt ... --out_dir runs/pvfm/eval_step4 --save_arrays
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from data import NormStats
from engine import CheckpointManager
from models import PVFlowMatcher
from visualize import Evaluator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: <ckpt-parent>/eval")
    p.add_argument("--max_sets", type=int, default=None)
    p.add_argument("--steps", type=int, default=1, help="Euler integration steps")
    p.add_argument("--chunk_pts", type=int, default=131072)
    p.add_argument("--use_ema", action="store_true",
                   help="evaluate the EMA shadow weights if the ckpt has them")
    p.add_argument("--save_arrays", action="store_true",
                   help="dump the (lf, hf, pred) cubes to .npz alongside plots")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    payload = CheckpointManager.load(args.ckpt, map_location=args.device)
    cfg = payload["cfg"]
    norm = NormStats.from_dict(payload["norm"])

    m = cfg["model"]
    default_c_env = 4 if cfg["data"].get("env_outside_mask", True) else 3
    model = PVFlowMatcher(
        c_pt=3, c_lf=3, c_env=m.get("c_env", default_c_env), c_lf_pt=3,
        n_style=m.get("n_style", 5),
        base_voxel=m.get("base_voxel", 32),
        base_point=m.get("base_point", 128),
        cond_dim=m.get("cond_dim", 256),
        n_blocks=m.get("n_blocks", 4),
        env_resolution=m.get("env_resolution", 64),
    ).to(args.device)

    state = payload["ema"] if (args.use_ema and payload.get("ema")) \
            else payload["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"[eval] loaded {'EMA' if args.use_ema else 'live'} weights "
          f"from {args.ckpt} (epoch {payload['epoch']})")

    out_dir = args.out_dir or str(Path(args.ckpt).parent / "eval")
    ev = Evaluator(model=model, cfg=cfg, norm=norm, out_dir=out_dir,
                   steps=args.steps, chunk_pts=args.chunk_pts)
    ev.run(max_sets=args.max_sets, save_arrays=args.save_arrays)


if __name__ == "__main__":
    main()
