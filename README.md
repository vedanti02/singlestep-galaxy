# StepOne-PVD — Single-Step Point-Voxel Flow Matching for HF/LF Galaxy Simulations

A fully supervised, single-step Point-Voxel Diffusion / Flow-Matching baseline that
predicts **high-fidelity** (HF) Lagrangian displacement fields from
**low-fidelity** (LF) inputs on the Quijote / Quijotelike datasets.

The implementation follows a specific algorithmic spec — overlapping cubic
regions of side `D` carrying *all* `D³` Lagrangian points, with the outer
`d/2` voxels excluded from the loss, conditioned on a **CNN summary of every
point outside the current region**.

---

## 1. Problem and approach

For each simulation set we have paired LF and HF cubes of Lagrangian
displacement (`disp.npy`, shape `(3, 64, 64, 64)` per tile). Sets are tiled
into ``(extent[0]·64) × (extent[1]·64) × (extent[2]·64)`` volumes, where
the per-set tile extent ranges from 1×1×1 to 6×6×6. A coarse 64³
"stitched" LF cube of the *whole* simulation is provided as a global
context view.

The model learns the **HF − LF residual** in normalized displacement
space using **conditional flow matching** (Lipman et al., 2023). At
inference we integrate the predicted velocity field with a single Euler
step (`steps=1`); larger `steps` trade compute for fidelity.

### Algorithm spec → code mapping

| Spec bullet | Implementation |
|---|---|
| Regions of size containing **up to n points** | Region side `D`; **every** one of `D³` cells is a point — no random subsampling. |
| **Overlap d in all three dimensions** | `crop_overlap`, applied via `overlap_crop_starts(L, D, overlap)` per axis. |
| **Ignore behaviour in external d/2** | Voxel `loss_mask` and per-cell `pt_mask` zero the outer `d/2` band; loss code consumes both. |
| **Conditioning on representation of all OTHER points** | Env voxels overlapping the current crop are **zeroed**, then a 4th indicator channel (1 outside, 0 inside) is appended. |
| **CNN-like average effect** | `GlobalContextEncoder` = 3D ConvNet → global average pool → 256-d token. |

---

## 2. Project layout

```
singlestep-galaxy/
├── config/
│   ├── __init__.py          # TypedDict schemas + load_config()
│   └── default.yaml         # default hyperparameters
│
├── ops/                     # pure spatial / spectral ops (no torch.nn / no I/O)
│   ├── geometry.py          # buffer crop, PBC, point ↔ voxel meshing,
│   │                        # outside_mask_for_crop
│   ├── spectrum.py          # P(k), T(k), coherence, cross-power
│   └── density.py           # CIC density deposit
│
├── data/                    # all I/O lives here, never imports models/
│   ├── readers.py           # NumpyTileReader, HDF5Reader stub + factory
│   ├── normalization.py     # NormStats + compute_norm_stats
│   ├── simulation_dataset.py# SimulationDataset (variable-extent crops)
│   ├── patch_collator.py    # PatchCollator
│   └── factory.py           # build_datasets / build_dataloaders / build_norm_stats
│
├── models/                  # network components (pure torch.nn)
│   ├── blocks.py            # ConvBlock3D, FiLMPointMLP, sinusoidal embed
│   ├── point_voxel_encoder.py    # local PV encoder + velocity head
│   ├── global_context_encoder.py # env CNN + style/time fusion
│   └── flow_matcher.py      # PVFlowMatcher wrapper
│
├── engine/                  # training / sampling logic
│   ├── flow_matching.py     # fm_targets, euler_sample
│   ├── losses.py            # masked_pt_mse, voxel_consistency_mse
│   ├── ema.py               # ModelEMA
│   ├── checkpoint.py        # CheckpointManager
│   └── trainer.py           # Trainer.fit / .validate / .resume
│
├── visualize.py             # Evaluator class (plots + JSON stats)
├── train.py                 # CLI: parse cfg → Trainer.fit
├── eval.py                  # CLI: load ckpt → Evaluator.run
├── train.sh                 # SLURM submission script
└── tests/
    └── test_geometry.py     # standalone tests of ops/geometry.py
```

### Decoupling rules

```
ops/        → numpy + torch         (no other internal deps)
config/     → stdlib + pyyaml       (no other internal deps)
data/       → ops/ + config/        (no models/, no engine/)
models/     → ops/ + config/        (no data/, no engine/)
engine/     → models/ + data/ + ops/ + config/
visualize.py→ data/ + models/ + ops/ + config/   (no engine/)
```

So `data/` and `ops/` are testable in isolation, and `models/` can be
swapped without touching I/O.

---

## 3. Data layout (read-only)

`/data/group_data/universedata/lagrangian_output_64/`

```
quijote-64/        HF tiles:  set{i}_pos_{x}_{y}_{z}/PART_009/{disp,vel,style}.npy
quijotelike-64/    LF tiles:  set{i}_pos_{x}_{y}_{z}/PART_009/{disp,vel,style}.npy
stitched/          full sim:  set{i}_quijote(like)/PART_009/{disp,vel,style}.npy   (64³ each)
```

* `disp.npy`  → `(3, 64, 64, 64)` Lagrangian displacement (x, y, z components).
* `vel.npy`   → `(3, 64, 64, 64)` velocity field (currently unused; v1 is disp-only).
* `style.npy` → `(5,)` cosmology vector (Ωₘ, Ωᵦ, h, nₛ, σ₈).

Per-set tile extent varies — most sets are 2×2×2 (128³), some 3³, 4³, 5³, 6³.
[`data/simulation_dataset.discover_sets`](data/simulation_dataset.py) only
keeps sets whose HF and LF bounding boxes are **complete** and that have a
matching stitched LF cube.

**Hold-out:** set IDs ending in `9` → test, ending in `8` → val, rest → train.

---

## 4. Configuration

Hyperparameters live in [`config/default.yaml`](config/default.yaml) and
are typed via the `TypedDict`s in [`config/__init__.py`](config/__init__.py).
Override on the CLI:

```bash
python train.py --override train.epochs=10 model.base_voxel=16
```

Key fields:

```yaml
data:
  crop_size: 32             # D — region side; per-crop point count = D**3 = 32_768
  crop_overlap: 8           # d — buffer per face = d/2 = 4 voxels
  env_outside_mask: true    # mask env to outside-of-crop + add indicator channel
  box_size: 1000.0          # Mpc/h (Quijote)
model:
  base_voxel: 32
  base_point: 128
  cond_dim: 256
  n_blocks: 4
  c_env: 4                  # 3 disp + 1 indicator (set to 3 if env_outside_mask=false)
optim: { lr: 2.0e-4, weight_decay: 1.0e-5, grad_clip: 1.0, ema_decay: 0.999 }
train: { epochs: 50, batch_size: 4, num_workers: 2, val_every: 1, ckpt_every: 5,
         device: cuda, out_dir: runs/pvfm }
flow:  { n_steps_train: 1, n_steps_infer: 1, lambda_voxel: 0.5 }
```

---

## 5. Architecture

```
                                                         style ⊕ sinusoidal(t)
stitched LF env (B, 4, 64, 64, 64)            ┌────────────────┐
  outside-masked + indicator   ──────────►    │ GlobalContextEnc│ ── g_env (B, 256)
                                              │   3D CNN +      │       │
                                              │   global pool   │       ▼ (sum)
                                              └────────────────┘   cond (B, 256)
LF voxel crop (B, 3, D, D, D) ───►  PointVoxelEncoder.lf_unet  ─►  lf_feat (B, 32, D, D, D)
                                                                                   │
   x_t (B, D³, 3) ─┐                                       trilinear gather       │
   coords (B, D³, 3)├──► concat → in_proj (1×1) → FiLMPointMLP × n_blocks ────────┘
   lf_pt  (B, D³, 3)│                              │ FiLM scale/shift = cond
                    └──────────────────────────────┘
                                    │
                                    ▼
                          head (1×1 conv) → v_θ (B, D³, 3)
```

* **Conditional flow matching:** for `t ~ U(0,1)`, build
  `x_t = (1−t) ε + t (HF−LF)` with `ε ~ N(0, I)`, train `v_θ` to predict
  `x_1 − x_0 = (HF−LF) − ε`. Single-step inference: `x ← ε + v_θ(ε, t=0)`.
* **Loss:** `masked_pt_mse(v_pred, v_target, pt_mask) + λ_vox · voxel_consistency_mse(...)`
  — see [`engine/losses.py`](engine/losses.py) for the docstring justifying
  MSE-on-residual (residual is approximately Gaussian and zero-mean;
  Chamfer / Poisson NLL would be wrong here).
* **EMA:** `ModelEMA(decay=0.999)` shadow weights, persisted in checkpoints.

---

## 6. Running

### Install

```bash
pip install torch numpy matplotlib pyyaml
```

(Optional: `h5py` if you ever wire up the HDF5 reader.)

### Train

```bash
# defaults from config/default.yaml
python train.py

# override an individual key
python train.py --override train.epochs=200 train.batch_size=8

# pick a different config file
python train.py --config config/my_run.yaml

# resume from a checkpoint (optimizer + EMA + epoch all restored)
python train.py --resume runs/pvfm/ckpt_latest.pt
```

Outputs land in `cfg.train.out_dir` (default `runs/pvfm/`):

```
runs/pvfm/
├── ckpt_latest.pt         # always points at the most recent ckpt
├── ckpt_epoch005.pt       # tagged snapshots
├── log.json               # per-epoch train / val metrics
└── eval/                  # populated by eval.py
```

### Evaluate

```bash
python eval.py --ckpt runs/pvfm/ckpt_latest.pt
python eval.py --ckpt runs/pvfm/ckpt_latest.pt --steps 4 --use_ema --max_sets 3
python eval.py --ckpt runs/pvfm/ckpt_latest.pt --save_arrays
```

For each held-out test set the [`Evaluator`](visualize.py) writes:

```
set{i}_disp_slice.png        # 2D mean intensity (LF / HF / Pred), per channel
set{i}_disp{c}_pk.png        # log-log P(k), one figure per channel
set{i}_disp_T.png            # transfer function T(k) across channels
set{i}_disp_r.png            # cross-coherence r(k) across channels
set{i}_disp_residual.png     # voxel-residual histogram with mean/median
set{i}_density_pk.png        # CIC density-field P(k)
set{i}_density_T_r.png       # density T(k) and r(k) on one panel
set{i}_density_slice.png     # mean-axis density slice (LF / HF / Pred)
set{i}_stats.json            # all binned k / P / T / r arrays + residual stats
```

### Tests

```bash
python tests/test_geometry.py        # standalone, no torch.nn / no I/O
# or with pytest
python -m pytest tests/test_geometry.py -v
```

---

## 7. SLURM submission

See [`train.sh`](train.sh) for a ready-made sbatch script. Submit with:

```bash
sbatch train.sh
```

It activates the user venv, sets thread caps, prints `nvidia-smi`, and runs
`train.py` with stdout/stderr piped to `logs/<JOBID>_*.{log,err}`. Add CLI
overrides at the bottom of the script.

---

## 8. Where to read the algorithm spec in code

| Spec point | File / function |
|---|---|
| Variable per-set extent + overlap-d crop schedule | [`data/simulation_dataset.py`](data/simulation_dataset.py) — `discover_sets`, `build_crop_index`, `SimulationDataset.__init__` |
| All D³ cells as points (no subsampling) | [`data/simulation_dataset.py`](data/simulation_dataset.py) — `_cell_idx`, `_cell_coords`, `__getitem__` |
| d/2 buffer mask | [`ops/geometry.py`](ops/geometry.py) — `inner_crop`, `edge_buffer_mask`, `point_in_inner_mask` |
| Outside-only env mask + indicator channel | [`ops/geometry.py`](ops/geometry.py) — `outside_mask_for_crop`; [`data/simulation_dataset.py`](data/simulation_dataset.py) — `_build_env`; mirrored in [`visualize.py`](visualize.py) |
| Local PV encoder | [`models/point_voxel_encoder.py`](models/point_voxel_encoder.py) — `_LFVoxelUNet`, `PointVoxelEncoder` |
| Global env encoder (CNN avg-effect) | [`models/global_context_encoder.py`](models/global_context_encoder.py) — `GlobalContextEncoder` |
| FM target + Euler sampler | [`engine/flow_matching.py`](engine/flow_matching.py) — `fm_targets`, `euler_sample` |
| Loss (rationale in docstring) | [`engine/losses.py`](engine/losses.py) — `masked_pt_mse`, `voxel_consistency_mse` |
| Trainer | [`engine/trainer.py`](engine/trainer.py) — `Trainer` |
| Power-spectrum / coherence baselines | [`ops/spectrum.py`](ops/spectrum.py) |

---

## 9. Extending

* **Add the velocity head.** Bump `c_pt` in
  [`models/flow_matcher.py`](models/flow_matcher.py) to 6 and concatenate
  velocity into the dataset's `tgt_pt`. Renormalize `vel` separately
  (its dynamic range is ≈ 100× displacement). The loss code is already
  generic over `c_pt`.

* **Add a new storage format.** Subclass
  [`data.readers.TileReader`](data/readers.py), implement `index_root`,
  `load_tile`, `load_full`, and register it in `_READERS`. The dataset
  picks it up via `cfg.data.reader = "your_name"`.

* **Variable region size.** Currently `D` is fixed; replace
  `build_crop_index` with a per-set `D_i` schedule and update
  `__getitem__` to size `_cell_idx` accordingly.

* **Strict outside-only env.** Already enabled by `data.env_outside_mask=true`.
  Disable to recover the global-env baseline (`c_env=3`).

---

## 10. Conventions

* Voxel grids: `(C, D, D, D)` (single) or `(B, C, D, D, D)` (batch).
* Point clouds: `(N, 3)` or `(B, N, 3)`.
* Coordinates passed to point↔voxel functions are normalized to `[0, 1]`.
* The simulation domain is a 3-torus of side `box_size` — see
  [`ops.geometry.apply_periodic_bc`](ops/geometry.py) and `minimum_image`.
* All disp values in the dataset are returned in **normalized** units
  (`(x − μ) / σ` with per-channel stats); denormalize via
  `NormStats.denormalize` for physical interpretation.
