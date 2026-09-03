# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Does making the affinity oracle differentiable help design small-molecule binders? A controlled study.
Gradient ascent through a differentiable Boltz-2 oracle reward-hacks and plateaus far below real crystal
binders, and the bottleneck is the pocket backbone, not the sequence. The differentiable counterpart to
NISE (Polizzi lab), which does the same job by gradient-free selection.

![The gradient folds a protein but forgets the pocket: the real binder buries the ligand, the de-novo design leaves it exposed](figures/hero.png)

- **It plateaus.** Across 14 ligands the best de novo design reaches held-out interface pTM 0.66 against
  0.93 for the real binders, and it beats gradient-free search (0.51 / 0.48) yet stalls at the same
  ceiling: the optimiser is not the limit.
- **The pocket is the limit.** Give the sequence layer a backbone, real or RFdiffusion3-generated, and it
  reaches 0.83 on a real backbone and 0.89 on RFdiffusion3 backbones across the panel, near the real
  binders. The gradient alone cannot build a foldable pocket and collapses to poly-alanine.
- **Reusable tooling.** A differentiable Boltz-2 affinity head (contributed to joltz); the
  differentiable inverse-folding term uses [jigandmpnn](https://github.com/nboyd/jigandmpnn) (Boyd).

The method, the controlled study, and a de novo protein-small-molecule affinity benchmark are written up
in the workshop paper (under review). This repo holds the code and the scored data behind every number.

## Install
The repo ships a `uv.lock`, so `uv` is the reliable path. It resolves the `git+https` deps
(`joltz`, `mosaic`, `jigandmpnn`) that plain pip struggles with:
```
uv sync                       # build .venv from the lockfile
# GPU JAX for mosaic/joltz on CUDA 12, into the same venv:
uv pip install "jax[cuda12]==0.10.1" nvidia-cudnn-cu12==9.17.0.29 \
   nvidia-cusolver-cu12==11.7.3.90 nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```
Plain `pip install -e .` also works, but has to resolve `mosaic` from git; if that fights you, use `uv`.

## Weights
Neither set of weights is stored in this repo. Both are standard third-party model files.

**Boltz-2** (needed for every fold). Into `~/.boltz`, or wherever `NISEGRAD_BOLTZ_CACHE` points (a single
knob that roots the checkpoints and the CCD cache together): the two model checkpoints `boltz2_conf.ckpt`
and `boltz2_aff.ckpt`, and the CCD chemistry `mols.tar` (extract it into a `mols/` directory). The `boltz`
CLI populates the cache on first run, but only when the conf checkpoint is missing, and it never re-fetches
the affinity checkpoint on its own, so when seeding the cache by hand include `boltz2_aff.ckpt`. `ccd.pkl`
(from the boltz-1 repo) is the old monolithic CCD; the Boltz-2 fold path reads `mols/` instead and never
opens it, so you can skip `ccd.pkl`.

**LigandMPNN** (needed *only* for `rescue_backbone.py`, `project.py`, `guided_design.py` — not for the
core STE panel or baselines, which need only Boltz-2). Run `bash scripts/get_weights.sh`, which fetches
the checkpoint into `weights/` and prints the env var to export. It comes from the official LigandMPNN
repo (`github.com/dauparas/LigandMPNN`) and is not on GitHub as a file. To do it by hand:
- `ligandmpnn_v_32_010_25.pt` from `files.ipd.uw.edu/pub/ligandmpnn/` (LigandMPNN's `get_model_params.sh`).
  Set `LIGANDMPNN_CKPT` to its path.

The differentiable JAX LigandMPNN itself is [jigandmpnn](https://github.com/nboyd/jigandmpnn) (Boyd), a
dependency that loads this checkpoint; it vendors the torch reference, so no separate reference module or
`LIGMPNN_MODEL_DIR` is needed.

**Protenix-v2** (the held-out judge; used by `protenix_score.py` and `panel_run.py`). The judge is a
separate ByteDance Protenix checkout with its own venv; the scripts run `$PROTENIX_DIR/.venv/bin/protenix
pred ... -n protenix-v2`, so point `PROTENIX_DIR` at that checkout. Two gotchas that cost real time:
- Build that venv on **Python 3.11**, not 3.12. Protenix ships prebuilt fused CUDA kernels compiled for
  cp311, which a 3.12 venv cannot import.
- The official HTTPS download for the `protenix-v2.pt` weights is currently blocked. Fetch it from the
  community mirror `huggingface.co/TMF001/protenix-v2-weights` instead.

## Reproducibility tiers
Not everything here is runnable from a clean machine. Two tiers:

- **From scratch.** The STE panel, the gradient-free baselines, the rescue and projector, and the affinity
  benchmark. Needs Boltz-2 and LigandMPNN (both reachable) plus the Protenix-v2 judge (a separate checkout,
  Python 3.11, weights via the mirror above). Every source above is live.
- **Data-only** (regeneration needs external installs we cannot redistribute): the RFdiffusion3 backbone
  generation (`panel_rfd3.sh`, `run2_rfd3_arm.sh`, pointed at an RFd3 pipeline via `CHEM_FAP_DIR`) and the
  RF2AA independence fold (`rf2aa_independence.py`, via `RF2AA_DIR` + a micromamba env). Their scored
  outputs are committed (`scripts/data/rfd3_panel/`, `rf2aa_scored.json`), so those numbers reproduce from
  data without rerunning.

Run `bash scripts/check_setup.sh` to see what is present and what is reachable before starting.

## Layout
- `src/nisegrad/` differentiable `P(bind)(sequence, ligand)` oracle and the STE optimiser
- `scripts/matched_budget.py` STE / Best-K-of-N / O3 designs at a fold budget
- `scripts/panel_baseline.py` gradient-free baseline over the ligand panel
- `scripts/rescue_backbone.py` design a fresh sequence on a frozen real pocket backbone
- `scripts/run2_rfd3_arm.sh` RFdiffusion3 backbone → LigandMPNN → Protenix
- `scripts/protenix_score.py` held-out Protenix judge
- `scripts/data/` the scored experiment logs behind every number: `panel_run/` (STE panel),
  `ste_best40/` (STE best-of-40 budget match), `rfd3_panel/` (backbone arm), `rescue_panel/`
  (real-backbone rescue; `rescue_panel_nocomp/` is the collapsed comp-weight-0 run, see its NOTE),
  `bench47.json` + `benchmark.json` (affinity benchmark), `rf2aa_scored.json` (independence judge)

## Credit
Builds on NISE (Polizzi lab), DBMol (Qin et al. 2026), BindEnergyCraft (Nori
et al. 2025), LambdaZero (Korablyov et al. 2024), LigandMPNN (Dauparas et al.), and RoseTTAFold All-Atom
(Krishna et al. 2024). Tools: Boltz-2 via [joltz](https://github.com/nboyd/joltz), Protenix, mosaic, and
[jigandmpnn](https://github.com/nboyd/jigandmpnn) (Boyd).
