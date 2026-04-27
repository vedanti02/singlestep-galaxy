"""End-to-end trainer that ties Config → data → model → losses → ckpt."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import (NormStats, build_dataloaders, build_datasets, get_reader)
from models import PVFlowMatcher

from .checkpoint import CheckpointManager
from .ema import ModelEMA
from .flow_matching import fm_targets
from .losses import masked_pt_mse, voxel_consistency_mse


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_model(cfg: Config) -> PVFlowMatcher:
    m = cfg["model"]
    # If env_outside_mask is on the env carries an extra indicator channel,
    # so c_env defaults to 4. The user can override via cfg["model"]["c_env"].
    default_c_env = 4 if cfg["data"].get("env_outside_mask", True) else 3
    return PVFlowMatcher(
        c_pt=3, c_lf=3, c_env=m.get("c_env", default_c_env), c_lf_pt=3,
        n_style=m.get("n_style", 5),
        base_voxel=m.get("base_voxel", 32),
        base_point=m.get("base_point", 128),
        cond_dim=m.get("cond_dim", 256),
        n_blocks=m.get("n_blocks", 4),
        env_resolution=m.get("env_resolution", 64),
    )


class Trainer:
    """High-level training driver.

    Public API:
        ``Trainer(cfg).fit()``              — train for ``cfg['train']['epochs']``.
        ``trainer.validate(loader)``        — one validation pass.
        ``trainer.resume(ckpt_path)``       — restore optimizer + EMA + epoch.
    """

    def __init__(self, cfg: Config,
                 model: Optional[PVFlowMatcher] = None,
                 norm: Optional[NormStats] = None) -> None:
        self.cfg = cfg
        self.device = cfg["train"].get("device", "cpu")
        _set_seed(cfg["train"].get("seed", 0))

        # data
        reader = get_reader(cfg["data"].get("reader", "numpy"))
        self.datasets, self.norm = build_datasets(cfg, reader=reader, norm=norm)
        self.loaders: dict[str, DataLoader] = build_dataloaders(cfg, self.datasets)

        # model
        self.model = (model or _build_model(cfg)).to(self.device)
        self.ema = ModelEMA(self.model,
                            decay=cfg["optim"].get("ema_decay", 0.999))

        # optim
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"].get("weight_decay", 1e-5),
        )
        self.scaler = torch.amp.GradScaler(
            enabled=str(self.device).startswith("cuda"))

        # bookkeeping
        out_dir = Path(cfg["train"].get("out_dir", "runs/pvfm"))
        self.ckpt = CheckpointManager(out_dir)
        self.out_dir = out_dir
        self.start_epoch = 0
        self.history: dict[str, list] = {"train": [], "val": []}

    # ------------------------------------------------------------------
    # core step
    # ------------------------------------------------------------------

    def _step(self, batch: dict) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        d = self.device
        lf_voxel = batch["lf_voxel"].to(d)
        env      = batch["env"].to(d)
        coords   = batch["coords"].to(d)
        lf_pt    = batch["lf_pt"].to(d)
        tgt_pt   = batch["tgt_pt"].to(d)
        tgt_vox  = batch["tgt_vox"].to(d)
        loss_mask = batch["loss_mask"].to(d)
        pt_mask   = batch["pt_mask"].to(d)
        style     = batch["style"].to(d)

        B = lf_voxel.shape[0]
        t = torch.rand(B, device=d)
        x_t, v_target = fm_targets(tgt_pt, t)

        lf_feat, cond = self.model.encode_cond(lf_voxel, env, style, t)
        v_pred = self.model(x_t, coords, lf_pt, lf_feat, cond)

        pt_loss = masked_pt_mse(v_pred, v_target, pt_mask)

        # x1 estimate from interpolant + velocity
        x1_hat = x_t + (1.0 - t.view(-1, 1, 1)) * v_pred
        vox_loss = voxel_consistency_mse(x1_hat, coords, tgt_vox, loss_mask)

        lam = cfg["flow"].get("lambda_voxel", 0.5)
        loss = pt_loss + lam * vox_loss
        return {"loss": loss,
                "pt_loss": pt_loss.detach(),
                "vox_loss": vox_loss.detach()}

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        bs = self.cfg["train"]["batch_size"]
        clip = self.cfg["optim"].get("grad_clip", 1.0)
        running = {"loss": 0.0, "pt_loss": 0.0, "vox_loss": 0.0, "n": 0}
        t0 = time.time()

        for step, batch in enumerate(self.loaders["train"]):
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                    device_type=str(self.device).split(":")[0],
                    enabled=str(self.device).startswith("cuda")):
                losses = self._step(batch)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.ema.update(self.model)

            running["loss"]    += float(losses["loss"]) * bs
            running["pt_loss"] += float(losses["pt_loss"]) * bs
            running["vox_loss"]+= float(losses["vox_loss"]) * bs
            running["n"]       += bs
            if step % 20 == 0:
                print(f"  e{epoch:03d} step {step:05d}  "
                      f"loss={float(losses['loss']):.4f} "
                      f"pt={float(losses['pt_loss']):.4f} "
                      f"vox={float(losses['vox_loss']):.4f}")

        n = max(running["n"], 1)
        out = {k: v / n for k, v in running.items() if k != "n"}
        out.update(epoch=epoch, dt=time.time() - t0)
        return out

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> dict[str, float]:
        if loader is None:
            loader = self.loaders.get("val")
        if loader is None or len(loader) == 0:
            return {}
        self.model.eval()
        bs = self.cfg["train"]["batch_size"]
        running = {"loss": 0.0, "pt_loss": 0.0, "vox_loss": 0.0, "n": 0}
        for batch in loader:
            losses = self._step(batch)
            running["loss"]    += float(losses["loss"]) * bs
            running["pt_loss"] += float(losses["pt_loss"]) * bs
            running["vox_loss"]+= float(losses["vox_loss"]) * bs
            running["n"]       += bs
        n = max(running["n"], 1)
        return {k: v / n for k, v in running.items() if k != "n"}

    # ------------------------------------------------------------------
    # public driver
    # ------------------------------------------------------------------

    def fit(self) -> None:
        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg["train"]["epochs"]):
            train_log = self._train_one_epoch(epoch)
            self.history["train"].append(train_log)
            print(f"[epoch {epoch}] train avg "
                  f"loss={train_log['loss']:.4f} pt={train_log['pt_loss']:.4f} "
                  f"vox={train_log['vox_loss']:.4f}  ({train_log['dt']:.1f}s)")

            if (epoch + 1) % cfg["train"].get("val_every", 1) == 0:
                v = self.validate()
                if v:
                    v["epoch"] = epoch
                    self.history["val"].append(v)
                    print(f"[epoch {epoch}]   val avg "
                          f"loss={v['loss']:.4f} pt={v['pt_loss']:.4f} "
                          f"vox={v['vox_loss']:.4f}")

            if (epoch + 1) % cfg["train"].get("ckpt_every", 5) == 0 \
                    or epoch == cfg["train"]["epochs"] - 1:
                self.ckpt.save(epoch=epoch, model=self.model,
                               optim=self.opt, norm=self.norm,
                               cfg=cfg, ema_state=self.ema.shadow_state_dict(),
                               tag=f"epoch{epoch:03d}")

            with open(self.out_dir / "log.json", "w") as f:
                json.dump(self.history, f, indent=2)

    def resume(self, ckpt_path: str) -> None:
        payload = CheckpointManager.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["optim"])
        if payload.get("ema") is not None:
            self.ema.load_shadow(payload["ema"])
        self.start_epoch = int(payload["epoch"]) + 1
        print(f"[resume] from epoch {self.start_epoch}")
