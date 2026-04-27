"""Training CLI entry point.

Usage:
    python train.py                                 # use config/default.yaml
    python train.py --config myrun.yaml             # override file
    python train.py --override train.epochs=10      # patch a single key
    python train.py --resume runs/pvfm/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
from typing import Any

from config import load_config
from engine import Trainer


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``--override key.subkey=value`` pairs into a nested dict."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--override must be key=value, got '{it}'")
        k, v = it.split("=", 1)
        # try numeric / bool
        for cast in (int, float):
            try:
                v_cast: Any = cast(v); break
            except ValueError:
                v_cast = v
        if isinstance(v_cast, str) and v_cast.lower() in ("true", "false"):
            v_cast = v_cast.lower() == "true"
        cur = out
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v_cast
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="path to a config YAML; defaults to config/default.yaml")
    p.add_argument("--override", nargs="*", default=[],
                   help="ad-hoc overrides, e.g. train.epochs=10")
    p.add_argument("--resume", type=str, default=None,
                   help="optional path to a ckpt_*.pt to resume from")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=_parse_overrides(args.override))
    trainer = Trainer(cfg)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
