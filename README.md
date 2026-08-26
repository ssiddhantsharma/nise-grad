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
- **Reusable tooling.** A differentiable Boltz-2 affinity head (contributed to joltz) and a
  parity-checked JAX LigandMPNN port ([jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn)).

The method, the controlled study, and a de novo protein-small-molecule affinity benchmark are written up
in the workshop paper (under review). This repo holds the code and the scored data behind every number.

## Install
```
pip install -e .
# GPU JAX that works with mosaic/joltz on CUDA 12:
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```
Checkpoints in `~/.boltz` (or set `NISEGRAD_BOLTZ_CACHE`): `boltz2_conf.ckpt`, `boltz2_aff.ckpt`.

## Layout
- `src/nisegrad/` differentiable `P(bind)(sequence, ligand)` oracle and the STE optimiser
- `scripts/matched_budget.py` STE / Best-K-of-N / O3 designs at a fold budget
- `scripts/panel_baseline.py` gradient-free baseline over the ligand panel
- `scripts/rescue_backbone.py` design a fresh sequence on a frozen real pocket backbone
- `scripts/run2_rfd3_arm.sh` RFdiffusion3 backbone → LigandMPNN → Protenix
- `scripts/protenix_score.py` held-out Protenix judge
- `scripts/data/` the scored experiment logs behind every number

## Credit
Builds on NISE (Polizzi lab), DRaFT (Clark et al. 2023), DBMol (Qin et al. 2026), BindEnergyCraft (Nori
et al. 2025), LambdaZero (Korablyov et al. 2024), LigandMPNN (Dauparas et al.), and RoseTTAFold All-Atom
(Krishna et al. 2024). Tools: Boltz-2 via [joltz](https://github.com/nboyd/joltz), Protenix, mosaic, and
[jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn).
