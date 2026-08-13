# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Design small-molecule-binding proteins by gradient descent: optimize the binder sequence
directly through a differentiable structure model (joltz / Boltz-2) and a differentiable
P(bind). A gradient counterpart to NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w), which does the same job with a
gradient-free selection loop. Not a fork.

## The result
Naive gradient ascent on the *soft* sequence reward-hacks: its optimum (soft P(bind) ~0.8) does
not transfer to a discrete sequence (discrete 0.1-0.3), and the designs are degenerate (poly-Leu
or poly-Phe). A **straight-through discrete step** fixes it: feed the oracle the discrete argmax
in the forward pass (so the objective is the real discrete P(bind)) and pass the gradient
through the soft distribution.

![gradient reality](figures/gradient_reality.png)

Measured on a 30aa binder vs benzoic acid (recycling=3, `num_sampling_steps=25` for physical
geometry):

| | discrete P(bind) | composition | fold |
|---|---|---|---|
| naive (soft) | 0.11-0.34 | degenerate (hyd 0.7-0.9) | |
| **straight-through** | **0.64** | **realistic (hyd 0.43)** | 119/119 bonds, pLDDT 0.79 |

```python
optimize_pbind(oracle, feats, 30, steps=25, recycling_steps=3,
               num_sampling_steps=25, straight_through=True)
```
This keeps NISE's key principle -- only score real discrete sequences -- but reaches a design in
~25 gradient steps (~25 recycled folds) instead of NISE's thousands. Caveats: single seed,
single ligand, P(bind) is the Boltz-2 oracle (not an experiment), and physical geometry needs
`nss>=25` (~25x slower than nss=2). `scripts/gradient_reality.py` regenerates the figure.

## Optional: ligand-aware prior
`src/nisegrad/ligand_mpnn_reg.py` adds a LigandMPNN
([jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn)) sequence prior as a drop-in
`mpnn=` term (`src/nisegrad/boltz_ligand.py` builds it from a Boltz feature dict + output),
useful for pushing composition or selectivity further.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)` (`recycling_steps`,
  `num_sampling_steps`)
- `src/nisegrad/optimize.py` gradient ascent: P(bind) (with `straight_through`), MPNN-regularized,
  or on-minus-off selectivity
- `src/nisegrad/ligand_mpnn_reg.py`, `boltz_ligand.py` the optional ligand-aware prior
- `scripts/` gradient check, optimization runs, and the figure

## Install
Built on joltz (with the affinity head) + mosaic. GPU:
```
pip install -e .
# GPU jax that works with mosaic/joltz on CUDA 12 (newer nvidia libs break cuSPARSE):
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```
Checkpoints (in `~/.boltz`, or point `NISEGRAD_BOLTZ_CACHE` at a dir with `ccd.pkl` + `mols/`):
`boltz2_conf.ckpt` (structure), `boltz2_aff.ckpt` (affinity head).

## Running / efficiency
- `straight_through=True` optimizes the discrete objective (use it).
- `num_sampling_steps`: 2 is fast but non-physical; >=25 for real geometry, ~25x slower.
- Independent designs are embarrassingly parallel: one per GPU (data-parallel / `pmap`) for
  campaign throughput. A single design is not easily split across GPUs; use gradient
  checkpointing for the `nss>=25` memory.
