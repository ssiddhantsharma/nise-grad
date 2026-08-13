# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Design small-molecule-binding proteins by gradient descent: optimize the binder sequence
directly through a differentiable structure model (joltz / Boltz-2) and a differentiable
P(bind). A gradient counterpart to NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w), which does the same job with a
gradient-free selection loop. Not a fork.

## What works
The machinery: P(bind) is differentiable end to end, the gradient of the Boltz-2 affinity head
flows to the soft binder sequence (`scripts/spike_affinity_gradient.py`), and
`optimize_pbind` ascends it.

## What doesn't (the honest result)
Naive gradient ascent on P(bind) **reward-hacks, and its soft optimum does not transfer to a
discrete sequence.** Measured on a 30aa binder vs benzoic acid (`figures/gradient_reality.png`,
`scripts/gradient_reality.py`):

![gradient reality](figures/gradient_reality.png)

- The optimizer drives the **soft** sequence to P(bind) ~0.8, but the P(bind) of the
  **discrete** argmax, refolded, is only 0.1-0.3. The continuous optimum sits between real
  sequences.
- The designs are degenerate: an all-hydrophobic string at `num_sampling_steps=2`, an
  all-aromatic (poly-Phe) string at `nss=25`.
- And the structure the affinity head scores is only physical (119/119 backbone bonds) at
  `nss>=25`; `nss=2` is noise. Physical geometry costs ~25x (6.7 vs 0.26 s/step at 30aa).

So gradient-through-Boltz optimizes an adversarial soft-sequence direction, not a real binder.
This is why **NISE spends its GPU on thousands of *discrete* LigandMPNN samples with full
folds**: only ever scoring real, discrete sequences structurally avoids these soft-sequence
optima. The open problem here is bridging the soft-to-discrete gap (a discrete/straight-through
estimator, or a sequence prior strong enough to keep the soft state near real sequences).

## Ligand-aware prior (a lever, not a fix)
`src/nisegrad/ligand_mpnn_reg.py` scores the soft sequence with LigandMPNN
([jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn)), a drop-in `mpnn=` term for
`optimize_pbind`. It pulls the sequence toward realistic composition, but the P(bind) gradient
is ~28x larger, so it only wins at a large weight (which then just inverse-folds the scaffold).
`src/nisegrad/boltz_ligand.py` builds it from a Boltz feature dict + output.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)` (`recycling_steps`,
  `num_sampling_steps` knobs)
- `src/nisegrad/optimize.py` gradient ascent: P(bind), MPNN-regularized, or on-minus-off
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
- `num_sampling_steps`: 2 is fast but non-physical; >=25 for real geometry, ~25x slower.
- `recycling_steps`: helps the trunk but does not make `nss=2` structures physical.
- Independent designs are embarrassingly parallel: one per GPU (data-parallel / `pmap`) for
  campaign throughput. A single design is not easily split across GPUs (would need model
  sharding); use gradient checkpointing for the `nss>=25` memory.
