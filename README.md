# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Design small-molecule-binding proteins by gradient descent: optimize the binder
sequence directly through a differentiable structure model (joltz / Boltz-2) and a
differentiable P(bind).

Reimplements the idea of NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w) with gradients instead of a
gradient-free selection loop. Not a fork.

## What works
- Differentiable P(bind): the gradient of the Boltz-2 affinity head flows to the
  (soft) binder sequence through the structure model
  (`scripts/spike_affinity_gradient.py`).
- Gradient optimization of the sequence, optionally regularized by a ProteinMPNN
  sequence log-likelihood (`src/nisegrad/optimize.py`).

## What doesn't (open problem)
Naive gradient ascent on P(bind) reward-hacks: it drives the sequence to a degenerate
hydrophobic string that maxes the affinity logit without being a real binder. A
ProteinMPNN log-likelihood regularizer keeps the sequence protein-like, but P(bind) then
does not improve and ProteinMPNN is *ligand-blind*, so it cannot see the interaction the
oracle is gaming.

## Ligand-aware regularizer
`src/nisegrad/ligand_mpnn_reg.py` scores the soft binder sequence with **LigandMPNN**
([jlig_mpnn](https://github.com/ssiddhantsharma/jlig_mpnn), a JAX port), conditioning on the
backbone *and* the ligand context. Its NLL is a drop-in `mpnn=` regularizer for
`optimize_pbind`, differentiable in the sequence. `src/nisegrad/boltz_ligand.py` maps a Boltz
feature dict + output into LigandMPNN inputs (verified against the real feature layout);
`scripts/optimize_pbind_ligandmpnn.py` runs it.

### The recipe (measured; see `figures/recipe_convergence.png`)
LigandMPNN is an inverse-folding prior, so it needs a *physical* backbone. Two findings got us
there:

1. **Fold a recycled scaffold, once, and freeze it.** The per-step diffusion structure at
   `recycling_steps=0` (the oracle default, for speed) is noise: ~0 of 119 backbone bonds at 2
   steps, still only ~20 at 200. `recycling_steps=3` gives 119/119. So fold the scaffold once
   with recycling (no gradient), freeze it, and score every step's sequence against it
   (`build_boltz_regularizer(..., frozen_output=scaffold)`). The geometry gate then passes at
   N-CA 1.44 A.
2. **Weight the prior above the P(bind) gradient.** At the start point `||grad -P(bind)||` is
   28x `||grad LigandMPNN||`, so a small weight is ignored (weight 2 and 10 both collapse to an
   all-hydrophobic string). Past the parity weight (~29) the prior wins: hydrophobic fraction
   drops 0.97 -> 0.77 -> 0.40 over weight 30 -> 60 -> 100, landing at LigandMPNN's own realistic
   preference. P(bind) falls as the weight rises: a strong structure prior drives the design
   toward inverse-folding the frozen scaffold rather than the binding logit, so scaffold quality
   is the next lever.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)`
- `src/nisegrad/optimize.py` gradient ascent: P(bind), MPNN-regularized, or on-minus-off selectivity
- `src/nisegrad/ligand_mpnn_reg.py` ligand-aware LigandMPNN sequence regularizer
- `scripts/` the gradient check and the optimization run

## Install
Built on joltz (with the affinity head) + mosaic. GPU:

```
pip install -e .
# GPU jax that works with mosaic/joltz on CUDA 12 (newer nvidia libs break cuSPARSE):
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```

Checkpoints (in `~/.boltz`, or point `NISEGRAD_BOLTZ_CACHE` at a dir with `ccd.pkl` +
`mols/`): `boltz2_conf.ckpt` (structure), `boltz2_aff.ckpt` (affinity head).

## Running efficiently
The cost is backprop through Boltz-2's diffusion structure module.
- `num_sampling_steps` (`PbindOracle`) is the main runtime/memory knob 2–8 steps give
  a usable gradient; more is slower and OOMs sooner.
- Binder length drives O(N²) memory; ~30–40 aa at 2–4 sampling steps fits a 48 GB GPU.
- For P(bind) alone (no MPNN regularizer) you only need `structure_coordinates`, so the
  confidence module can be skipped for a further memory/speed cut.
- Larger designs need gradient checkpointing on the diffusion.
