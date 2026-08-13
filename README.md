# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Design small-molecule-binding proteins by gradient descent: optimize the binder sequence
directly through a differentiable structure model (Boltz-2 via joltz) and a differentiable
P(bind). A gradient counterpart to NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w), which does the same job with a
gradient-free selection loop. Not a fork.

## The problem: naive gradient reward-hacks
Optimizing the *soft* (continuous) sequence maximizes a fiction. Its optimum sits between real
amino acids, so the soft P(bind) climbs to ~0.8, but the discrete argmax sequence, refolded,
scores only 0.1-0.3, and the design is degenerate (poly-Leu / poly-Phe).

## The fix: a straight-through estimator (STE)
The argmax that turns the soft distribution into a real sequence is not differentiable, so you
cannot backprop through "pick the top amino acid." STE sidesteps this: **score the discrete
sequence in the forward pass, but let the gradient flow through the soft distribution.** Each
step, take the one-hot `argmax(softmax(logits))` and fold *that* (so the objective is the real
discrete P(bind)); in the backward pass, treat the one-hot as if it were the softmax:

```python
hard = one_hot(argmax(soft))
seq  = soft + stop_gradient(hard - soft)   # forward = hard, backward = soft
```

Now the value being maximized is the discrete design's P(bind), not a soft illusion, while the
logits still get a gradient. Enable it with `optimize_pbind(..., straight_through=True)`.

![gradient reality](figures/gradient_reality.png)

*30aa binder, recycling=3, `num_sampling_steps=25`, 3 seeds each. Naive's soft P(bind) reaches
~0.9, but the discrete design scores 0.33-0.41 and is degenerate. STE optimizes the discrete
design directly: 0.44-0.73 with realistic composition, well-folded (119/119 backbone bonds,
pLDDT ~0.8).*

This keeps NISE's principle -- only ever score real discrete sequences -- in ~25 gradient steps
instead of NISE's thousands of folds. Caveats: P(bind) is the Boltz-2 oracle, not an experiment;
harder ligands are more variable (sulfonamide 0.44-0.66 vs benzoic acid 0.62-0.73); `nss>=25` is
~25x slower than `nss=2`. Runnable: `scripts/optimize_ste.py`.

## Optional: ligand-aware prior
`src/nisegrad/ligand_mpnn_reg.py` adds a LigandMPNN
([jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn)) sequence prior as a drop-in
`mpnn=` term (built from a Boltz output by `src/nisegrad/boltz_ligand.py`), for pushing
composition or selectivity further.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)`
- `src/nisegrad/optimize.py` gradient ascent (`straight_through`, MPNN-regularized, selectivity)
- `src/nisegrad/ligand_mpnn_reg.py`, `boltz_ligand.py` optional ligand-aware prior
- `scripts/` gradient check, optimization runs, figure

## Install
```
pip install -e .
# GPU jax that works with mosaic/joltz on CUDA 12 (newer nvidia libs break cuSPARSE):
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```
Checkpoints in `~/.boltz` (or set `NISEGRAD_BOLTZ_CACHE`): `boltz2_conf.ckpt`, `boltz2_aff.ckpt`.
Independent designs are embarrassingly parallel (one per GPU) for campaign throughput.
