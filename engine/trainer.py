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
from models import PatchCritic3D, PVFlowMatcher

from .checkpoint import CheckpointManager
from .ema import ModelEMA
from .flow_matching import fm_targets, lf_init_sample
from .losses import (divergence_mse, gradient_penalty, masked_pt_mse,
                     voxel_consistency_mse)
from ops.density import cic_density
from ops.geometry import inner_crop, points_to_voxel
from ops.spectrum import coherence, transfer_function


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_model(cfg: Config) -> PVFlowMatcher:
    return PVFlowMatcher.from_config(cfg)


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
        self.datasets, self.norm, self.extra_norms = build_datasets(
            cfg, reader=reader, norm=norm)
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
        # Optional LR schedule: linear warmup then cosine decay.
        # Enable with cfg.optim.lr_schedule = "warmup_cosine".
        # cfg.optim.warmup_steps controls warmup length (default 500).
        sched_name = cfg["optim"].get("lr_schedule", "constant")
        if sched_name == "warmup_cosine":
            warmup = int(cfg["optim"].get("warmup_steps", 500))
            steps_per_epoch = max(1, len(self.datasets["train"]) //
                                  cfg["train"]["batch_size"])
            total_steps = max(1, steps_per_epoch * cfg["train"]["epochs"])

            def _lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                # cosine from 1.0 to 0.0 over the remaining steps
                progress = (step - warmup) / max(1, total_steps - warmup)
                progress = min(max(progress, 0.0), 1.0)
                import math
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=_lr_lambda)
            self._global_step = 0
        else:
            self.lr_sched = None
            self._global_step = 0
        # torch.cuda.amp.* is available since 1.6; the newer torch.amp.*
        # API only exists in 2.5+, so use the legacy path for portability.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=str(self.device).startswith("cuda"))

        # Normalization constants as device tensors: used by the GAN step
        # and the spectral monitor for fp32 denorm -> physical -> voxel-
        # unit conversions.
        def _dev(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32,
                                   device=self.device).view(1, 3, 1, 1, 1)
        self._std_t = _dev(self.norm.std)
        self._mean_t = _dev(self.norm.mean)
        self._res_std_t = _dev(self.norm.res_std)
        self._res_mean_t = _dev(self.norm.res_mean)
        self._buf = int(cfg["data"]["crop_overlap"]) // 2
        self._box_size = float(cfg["data"].get("box_size", 1000.0))

        # ---- optional adversarial training (conditional WGAN-GP) ----
        gan_cfg = dict(cfg.get("gan") or {})
        self.gan_on = bool(gan_cfg.get("enabled", False))
        if self.gan_on:
            if cfg["flow"].get("mode") != "lf_init":
                raise ValueError("gan.enabled currently requires "
                                 "flow.mode == 'lf_init'")
            self.gan_cfg = gan_cfg
            n_style = cfg["model"].get("n_style", 5)
            self._d_style = bool(gan_cfg.get("d_style_proj", False))
            style_dim = n_style if self._d_style else 0
            d_base = int(gan_cfg.get("d_base", 64))
            # D_disp sees residual (3) + LF disp (3) conditioning; extra
            # LF fields (vel) are left out — disp carries the structure.
            self.d_disp = PatchCritic3D(c_in=6, base=d_base,
                                        style_dim=style_dim).to(self.device)
            self.d_dens = PatchCritic3D(c_in=2, base=d_base,
                                        style_dim=style_dim).to(self.device)
            self._d_params = (list(self.d_disp.parameters())
                              + list(self.d_dens.parameters()))
            self.opt_d = torch.optim.Adam(
                self._d_params,
                lr=float(gan_cfg.get("d_lr", 1e-4)),
                betas=tuple(gan_cfg.get("d_betas", (0.0, 0.9))))
            self._dens_eps = float(gan_cfg.get("density_eps", 1e-6))
            self._n_critic = int(gan_cfg.get("n_critic", 1))
            self._gp_lambda = float(gan_cfg.get("gp_lambda", 10.0))

        # bookkeeping
        out_dir = Path(cfg["train"].get("out_dir", "runs/pvfm"))
        self.ckpt = CheckpointManager(out_dir)
        self.out_dir = out_dir
        self.start_epoch = 0
        self.history: dict[str, list] = {"train": [], "val": []}

    # ------------------------------------------------------------------
    # core step
    # ------------------------------------------------------------------

    def _step(self, batch: dict, train: bool = False) -> dict[str, torch.Tensor]:
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
        mode = cfg["flow"].get("mode", "flow_matching")
        gan_pack = None

        if mode == "direct":
            # Direct residual regression: model predicts tgt_pt itself.
            # x_t is fed random noise (not the placeholder zero) so the
            # model is forced to use the conditioning rather than fitting
            # a fixed input pattern. t stays 0.
            t = torch.zeros(B, device=d)
            x_t = torch.randn_like(tgt_pt)
            lf_feat, cond = self.model.encode_cond(lf_voxel, env, style, t)
            x1_pred = self.model(x_t, coords, lf_pt, lf_feat, cond)
            pt_loss = masked_pt_mse(x1_pred, tgt_pt, pt_mask)
            # In direct mode each cell has exactly one point, so the
            # voxel-scatter is the identity — vox_loss == pt_loss and
            # adding it just doubles the gradient. Skip it.
            vox_loss = torch.zeros((), device=d)
        elif mode == "lf_init":
            # LF-init flow matching: x_0 = LF disp (in normalized space),
            # x_1 = HF = LF + tgt_pt. The interpolant x_t = (1-t)*LF + t*HF
            # always lives in the data manifold (no random noise floor).
            # The analytic velocity is constant: v* = HF - LF = tgt_pt.
            # Optional t-bias: when t_alpha > 0, sample t ~ Beta(t_alpha, 1)
            # which biases toward small t. Inference uses t=0, and the model
            # cannot exploit the v = (x_t - lf_pt)/t shortcut at small t,
            # so this trains the conditioning-only path more thoroughly.
            t_alpha = cfg["flow"].get("t_alpha", 0.0)
            p_zero  = cfg["flow"].get("p_zero", 0.0)
            noise_sigma = cfg["flow"].get("noise_sigma", 0.0)
            if t_alpha > 0:
                u = torch.rand(B, device=d).clamp(min=1e-6)
                t = u.pow(1.0 / t_alpha)               # ~ Beta(t_alpha, 1)
            else:
                t = torch.rand(B, device=d)
            if p_zero > 0:
                # Force t=0 on a fraction of the batch — the model literally
                # trains on the inference distribution.
                t = torch.where(torch.rand(B, device=d) < p_zero,
                                torch.zeros_like(t), t)
            lf_disp_pt = lf_pt[..., :3]               # disp channels only
            x_1 = lf_disp_pt + tgt_pt                  # HF (normalized)
            t_ = t.view(-1, 1, 1)
            x_t = (1.0 - t_) * lf_disp_pt + t_ * x_1
            if noise_sigma > 0:
                # Inject Gaussian noise on x_t scaled by sqrt(t*(1-t)).
                # Vanishes at endpoints (so x_0=lf, x_1=hf are unchanged
                # in expectation), peaks in the middle. Breaks the
                # (x_t - lf_pt)/t shortcut without changing the optimal
                # constant velocity v* = HF - LF.
                scale = noise_sigma * torch.sqrt(t_ * (1.0 - t_) + 1e-8)
                x_t = x_t + scale * torch.randn_like(x_t)
            v_target = tgt_pt                          # constant velocity
            # Encode LF U-Net and env CNN once per batch; both the
            # random-t token here and the adversarial t=0 token below
            # reuse them (only the cheap style/time MLP + z differ).
            lf_feat = self.model.local.encode_voxel(lf_voxel)
            g_env = self.model.globalc.encode_env(env)
            cond = self.model.cond_token(env, style, t, g_env=g_env)
            v_pred = self.model(x_t, coords, lf_pt, lf_feat, cond)
            pt_loss = masked_pt_mse(v_pred, v_target, pt_mask)
            # Predicted residual at points (x_1 estimate minus LF) and
            # apply voxel-consistency on it against the residual cube.
            x1_hat = x_t + (1.0 - t_) * v_pred
            residual_pt_hat = x1_hat - lf_disp_pt
            vox_loss = voxel_consistency_mse(residual_pt_hat, coords,
                                             tgt_vox, loss_mask)
            if self.gan_on and train:
                # Dedicated t=0 forward for the adversarial path: this is
                # EXACTLY the single-step inference computation (x = LF,
                # t = 0, v = residual), so the critics shape the
                # distribution the paper evaluates. Reuses the cached
                # lf_feat and g_env; only the style/time MLP is recomputed
                # — with a fresh z sample when z_dim > 0.
                t0 = torch.zeros(B, device=d)
                cond0 = self.model.cond_token(env, style, t0, g_env=g_env)
                v0 = self.model(lf_disp_pt, coords, lf_pt, lf_feat, cond0)
                D_ = tgt_vox.shape[-1]
                # coords are the C-order raveled ij-meshgrid of the crop,
                # so (B, D^3, 3) -> (B, 3, D, D, D) is a lossless reshape.
                fake_res_cube = (v0.reshape(B, D_, D_, D_, 3)
                                 .permute(0, 4, 1, 2, 3))
                gan_pack = {
                    "fake_res_cube": fake_res_cube,
                    "tgt_vox": tgt_vox,
                    "lf_disp_cube": lf_voxel[:, :3],
                    "style": style,
                    "extent": batch["extent"],
                }
        else:
            t = torch.rand(B, device=d)
            x_t, v_target = fm_targets(tgt_pt, t)
            lf_feat, cond = self.model.encode_cond(lf_voxel, env, style, t)
            v_pred = self.model(x_t, coords, lf_pt, lf_feat, cond)
            pt_loss = masked_pt_mse(v_pred, v_target, pt_mask)
            # x1 estimate from interpolant + velocity
            x1_hat = x_t + (1.0 - t.view(-1, 1, 1)) * v_pred
            vox_loss = voxel_consistency_mse(x1_hat, coords, tgt_vox, loss_mask)

        lam = cfg["flow"].get("lambda_voxel", 0.5)
        lam_div = cfg["flow"].get("lambda_div", 0.0)

        # Auxiliary divergence loss matches first-order density via
        # delta ≈ -∇·u. Needs the predicted residual scattered to the
        # crop's voxel grid; we reuse the points_to_voxel rasterizer.
        div_loss = torch.zeros((), device=d)
        if lam_div > 0:
            # Predicted residual at points depends on mode:
            #   FM:       residual ≈ v_pred * (1 - 0) at t=0 inference, but
            #             we approximate using x1_hat - x_t at training t.
            #             Use voxel-form: scatter x1_hat to grid, subtract LF disp cube.
            #   direct:   x1_pred is the residual itself.
            #   lf_init:  v_pred is the residual (constant velocity = tgt).
            if mode == "direct":
                pred_pt = x1_pred
            elif mode == "lf_init":
                pred_pt = v_pred
            else:
                # x1_hat - x_t = (1-t)*v_pred (approximation of residual)
                pred_pt = (1.0 - t.view(-1, 1, 1)) * v_pred
            D_ = tgt_vox.shape[-1]
            pred_vox = points_to_voxel(coords, pred_pt, R=D_, reduction="mean")
            div_loss = divergence_mse(pred_vox, tgt_vox, loss_mask)

        loss = pt_loss + lam * vox_loss + lam_div * div_loss
        return {"loss": loss,
                "pt_loss": pt_loss.detach(),
                "vox_loss": vox_loss.detach(),
                "div_loss": div_loss.detach(),
                "gan_pack": gan_pack}

    # ------------------------------------------------------------------
    # adversarial machinery (conditional WGAN-GP, fp32 critics)
    # ------------------------------------------------------------------

    def _adv_lambdas(self, epoch: int) -> tuple[float, float]:
        """Per-epoch adversarial weights with a linear warmup ramp."""
        g = self.gan_cfg
        start = int(g.get("adv_ramp_start_epoch", 1))
        ramp = max(1, int(g.get("adv_ramp_epochs", 2)))
        s = min(max((epoch - start) / ramp, 0.0), 1.0)
        return (s * float(g.get("lambda_adv_disp", 5e-3)),
                s * float(g.get("lambda_adv_dens", 5e-3)))

    def _critic_inputs(self, pack: dict) -> dict:
        """Build fp32, inner-cropped critic inputs from a _step gan_pack.

        Displacement branch works in normalized units (scale-free). The
        density branch converts to physical units, then voxel units with
        the per-sample grid spacing dx = box / extent, CIC-deposits, and
        log-transforms. Everything is inner-cropped so the critics never
        see the crop-boundary band (mirroring pt_mask/loss_mask).
        """
        buf = self._buf
        fake = pack["fake_res_cube"].float()
        real = pack["tgt_vox"].float()
        lf_disp = pack["lf_disp_cube"].float()
        style = pack["style"].float() if self._d_style else None

        dx = torch.tensor([self._box_size / float(e[0]) for e in pack["extent"]],
                          device=fake.device, dtype=torch.float32
                          ).view(-1, 1, 1, 1, 1)
        lf_phys = lf_disp * self._std_t + self._mean_t
        res_real_phys = real * self._res_std_t + self._res_mean_t
        res_fake_phys = fake * self._res_std_t + self._res_mean_t

        # One batched deposit for (LF, real HF, fake HF): same FLOPs as
        # three calls but one meshgrid + one scatter pass. Only the fake
        # slice may carry gradient (LF/real are loader data) — but the
        # concatenated deposit puts all three slices in ONE autograd
        # graph, so the LF/real slices MUST be detached: otherwise the
        # critic update's backward (through D(real)) frees the shared
        # graph that the generator's later backward still needs.
        B = fake.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            rho = cic_density(torch.cat(
                [lf_phys / dx,
                 (lf_phys + res_real_phys) / dx,
                 (lf_phys + res_fake_phys) / dx], dim=0))
        log_rho = torch.log(rho + self._dens_eps)
        log_lf, log_real, log_fake = log_rho.split(B, dim=0)
        log_lf = log_lf.detach()
        log_real = log_real.detach()

        def ic(x):
            return inner_crop(x, buf)

        return {
            "disp_real": torch.cat([ic(real), ic(lf_disp)], dim=1),
            "disp_fake": torch.cat([ic(fake), ic(lf_disp)], dim=1),
            "dens_real": torch.cat([ic(log_real), ic(log_lf)], dim=1),
            "dens_fake": torch.cat([ic(log_fake), ic(log_lf)], dim=1),
            "style": style,
        }

    def _gan_step(self, pack: dict, clip: float) -> dict:
        """One critic update round + adversarial G terms.

        Runs entirely in fp32 (outside autocast): the WGAN-GP
        double-backward must not meet GradScaler-scaled gradients, and the
        critics are small enough that fp32 costs nothing. Returns the
        grad-attached adversarial G losses plus detached logs.
        """
        ins = self._critic_inputs(pack)
        # PatchCritic3D ignores its style arg when style_dim == 0, so it
        # is always safe to pass ins["style"] (None or tensor) through.
        style = ins["style"]
        log_t: dict[str, torch.Tensor] = {}

        # ---- critic updates (fakes detached) ----
        with torch.cuda.amp.autocast(enabled=False):
            for _ in range(self._n_critic):
                self.opt_d.zero_grad(set_to_none=True)
                d_loss_total = 0.0
                for name, critic, r_in, f_in in (
                        ("disp", self.d_disp, ins["disp_real"],
                         ins["disp_fake"].detach()),
                        ("dens", self.d_dens, ins["dens_real"],
                         ins["dens_fake"].detach())):
                    s_real = critic(r_in, style).mean()
                    s_fake = critic(f_in, style).mean()
                    gp = gradient_penalty(critic, r_in, f_in, style)
                    d_loss = s_fake - s_real + self._gp_lambda * gp
                    d_loss_total = d_loss_total + d_loss
                    log_t[f"d_{name}_loss"] = d_loss.detach()
                    log_t[f"gp_{name}"] = gp.detach()
                    log_t[f"w_{name}"] = (s_real - s_fake).detach()
                d_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self._d_params, clip)
                self.opt_d.step()

            # ---- adversarial G terms (critics frozen) ----
            self._set_critic_grads(False)
            adv_disp = -self.d_disp(ins["disp_fake"], style).mean()
            adv_dens = -self.d_dens(ins["dens_fake"], style).mean()
            self._set_critic_grads(True)

        log_t["adv_disp"] = adv_disp.detach()
        log_t["adv_dens"] = adv_dens.detach()
        # One GPU->CPU sync for all log scalars instead of one per value.
        keys = list(log_t)
        vals = torch.stack([log_t[k] for k in keys]).tolist()
        return {"adv_disp": adv_disp, "adv_dens": adv_dens,
                "logs": dict(zip(keys, vals))}

    def _set_critic_grads(self, flag: bool) -> None:
        for p in self._d_params:
            p.requires_grad_(flag)

    @torch.no_grad()
    def _spectral_val(self, epoch: int, n_batches: int) -> dict[str, float]:
        """Crop-level spectral monitor: mean |T-1| and coherence r on val.

        Uses the EMA weights and the exact single-step inference path
        (lf_init_sample, steps=1), then compares CIC densities of the
        predicted vs true HF field on the inner crop. Crop-level spectra
        are noisier than the full-cube eval but track the same failure
        mode epoch-by-epoch. Monitor only — never a loss.
        """
        loader = self.loaders.get("val")
        if loader is None:
            return {}
        model = self.ema.shadow
        model.eval()
        buf = self._buf
        T_errs, rs = [], []
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            d = self.device
            lf_voxel = batch["lf_voxel"].to(d)
            env = batch["env"].to(d)
            coords = batch["coords"].to(d)
            lf_pt = batch["lf_pt"].to(d)
            style = batch["style"].to(d)
            tgt_vox = batch["tgt_vox"].to(d).float()
            pred = lf_init_sample(model, lf_voxel, env, style,
                                  coords, lf_pt, steps=1)          # (B, N, 3)
            B = pred.shape[0]
            D_ = tgt_vox.shape[-1]
            pred_cube = (pred.float().reshape(B, D_, D_, D_, 3)
                         .permute(0, 4, 1, 2, 3))
            lf_disp = lf_voxel[:, :3].float()
            lf_phys = lf_disp * self._std_t + self._mean_t
            dx = torch.tensor(
                [self._box_size / float(e[0]) for e in batch["extent"]],
                device=d, dtype=torch.float32).view(-1, 1, 1, 1, 1)
            hf_true = lf_phys + (tgt_vox * self._res_std_t + self._res_mean_t)
            hf_pred = lf_phys + (pred_cube * self._res_std_t + self._res_mean_t)
            rho_t = inner_crop(cic_density(hf_true / dx), buf)[:, 0]
            rho_p = inner_crop(cic_density(hf_pred / dx), buf)[:, 0]
            for b in range(B):
                dt = (rho_t[b] - 1.0).cpu().numpy()
                dp = (rho_p[b] - 1.0).cpu().numpy()
                k, T, _, _ = transfer_function(dp, dt, n_bins=8)
                _, r = coherence(dp, dt, n_bins=8)
                lo = max(1, len(k) // 5)
                hi = max(lo + 1, 4 * len(k) // 5)
                T_errs.append(float(np.mean(np.abs(T[lo:hi] - 1.0))))
                rs.append(float(np.mean(r[lo:hi])))
        if not T_errs:
            return {}
        return {"spec_T_err": float(np.mean(T_errs)),
                "spec_r": float(np.mean(rs))}

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        bs = self.cfg["train"]["batch_size"]
        clip = self.cfg["optim"].get("grad_clip", 1.0)
        running = {"loss": 0.0, "pt_loss": 0.0, "vox_loss": 0.0, "n": 0}
        gan_running: dict[str, float] = {}
        if self.gan_on:
            lam_disp, lam_dens = self._adv_lambdas(epoch)
        t0 = time.time()

        for step, batch in enumerate(self.loaders["train"]):
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                    enabled=str(self.device).startswith("cuda")):
                losses = self._step(batch, train=True)
            total = losses["loss"]
            gan_logs: dict[str, float] = {}
            if self.gan_on and losses.get("gan_pack") is not None:
                # Critic update + adversarial G terms, all in fp32. The
                # critics train from epoch 0 (calibrated before the ramp
                # switches the generator term on).
                gan_out = self._gan_step(losses["gan_pack"], clip)
                total = (total
                         + lam_disp * gan_out["adv_disp"]
                         + lam_dens * gan_out["adv_dens"])
                gan_logs = gan_out["logs"]
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.lr_sched is not None:
                self.lr_sched.step()
            self._global_step += 1
            self.ema.update(self.model)

            running["loss"]    += float(losses["loss"]) * bs
            running["pt_loss"] += float(losses["pt_loss"]) * bs
            running["vox_loss"]+= float(losses["vox_loss"]) * bs
            running["n"]       += bs
            for k, v in gan_logs.items():
                gan_running[k] = gan_running.get(k, 0.0) + v * bs
            if step % 20 == 0:
                lr_now = self.opt.param_groups[0]["lr"]
                msg = (f"  e{epoch:03d} step {step:05d}  "
                       f"loss={float(losses['loss']):.4f} "
                       f"pt={float(losses['pt_loss']):.4f} "
                       f"vox={float(losses['vox_loss']):.4f} "
                       f"|g|={float(grad_norm):.3f} "
                       f"lr={lr_now:.2e}")
                if gan_logs:
                    msg += (f" | w_disp={gan_logs['w_disp']:.3f} "
                            f"w_dens={gan_logs['w_dens']:.3f} "
                            f"gp={gan_logs['gp_disp']:.2f}/"
                            f"{gan_logs['gp_dens']:.2f} "
                            f"lam={lam_disp:.1e}")
                print(msg)

        n = max(running["n"], 1)
        out = {k: v / n for k, v in running.items() if k != "n"}
        out.update({k: v / n for k, v in gan_running.items()})
        if self.gan_on:
            out.update(lam_adv_disp=lam_disp, lam_adv_dens=lam_dens)
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
        # Optional cap (train.max_val_batches) — smoke tests / CPU runs.
        maxb = int(self.cfg["train"].get("max_val_batches", 0) or 0)
        running = {"loss": 0.0, "pt_loss": 0.0, "vox_loss": 0.0, "n": 0}
        for bi, batch in enumerate(loader):
            if maxb and bi >= maxb:
                break
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
                n_spec = int(cfg["train"].get("spectral_monitor_batches", 0))
                if n_spec > 0:
                    v.update(self._spectral_val(epoch, n_spec))
                if v:
                    v["epoch"] = epoch
                    self.history["val"].append(v)
                    msg = (f"[epoch {epoch}]   val avg "
                           f"loss={v['loss']:.4f} pt={v['pt_loss']:.4f} "
                           f"vox={v['vox_loss']:.4f}")
                    if "spec_T_err" in v:
                        msg += (f"  |T-1|={v['spec_T_err']:.3f} "
                                f"r={v['spec_r']:.3f}")
                    print(msg)

            if (epoch + 1) % cfg["train"].get("ckpt_every", 5) == 0 \
                    or epoch == cfg["train"]["epochs"] - 1:
                disc_state = disc_optim = None
                if self.gan_on:
                    disc_state = {"d_disp": self.d_disp.state_dict(),
                                  "d_dens": self.d_dens.state_dict()}
                    disc_optim = self.opt_d.state_dict()
                self.ckpt.save(epoch=epoch, model=self.model,
                               optim=self.opt, norm=self.norm,
                               cfg=cfg, ema_state=self.ema.shadow_state_dict(),
                               tag=f"epoch{epoch:03d}",
                               extra_norms=self.extra_norms,
                               disc_state=disc_state,
                               disc_optim=disc_optim)

            with open(self.out_dir / "log.json", "w") as f:
                json.dump(self.history, f, indent=2)

    def resume(self, ckpt_path: str) -> None:
        payload = CheckpointManager.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["optim"])
        if payload.get("ema") is not None:
            self.ema.load_shadow(payload["ema"])
        if self.gan_on and payload.get("disc_state") is not None:
            self.d_disp.load_state_dict(payload["disc_state"]["d_disp"])
            self.d_dens.load_state_dict(payload["disc_state"]["d_dens"])
            if payload.get("disc_optim") is not None:
                self.opt_d.load_state_dict(payload["disc_optim"])
        self.start_epoch = int(payload["epoch"]) + 1
        print(f"[resume] from epoch {self.start_epoch}")
