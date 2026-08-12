# nise-grad

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
does not improve. Note ProteinMPNN is ligand-blind, which likely contributes to the
collapse; a ligand-aware model (LigandMPNN) used as a snap/generator on the predicted
pocket, then re-folded to score, is the intended next step.

## Layout
- `src/nisegrad/oracle.py` — differentiable `P(bind)(sequence, ligand)`
- `src/nisegrad/optimize.py` — gradient ascent: P(bind), MPNN-regularized, or on-minus-off selectivity
- `scripts/` — the gradient check and the optimization run

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
- `num_sampling_steps` (`PbindOracle`) is the main runtime/memory knob — 2–8 steps give
  a usable gradient; more is slower and OOMs sooner.
- Binder length drives O(N²) memory; ~30–40 aa at 2–4 sampling steps fits a 48 GB GPU.
- For P(bind) alone (no MPNN regularizer) you only need `structure_coordinates`, so the
  confidence module can be skipped for a further memory/speed cut.
- Larger designs need gradient checkpointing on the diffusion.
