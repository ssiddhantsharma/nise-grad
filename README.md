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
hydrophobic string that maxes the affinity logit without being a real binder. Adding a
ProteinMPNN log-likelihood regularizer keeps the sequence protein-like, but then P(bind)
does not improve — the folded structure of these small de-novo binders collapses, so
the affinity head and MPNN both prefer hydrophobic. Producing real binders needs more
than a soft regularizer; that is the next research step, not assembly.

## Layout
- `src/nisegrad/oracle.py` — differentiable `P(bind)(sequence, ligand)`
- `src/nisegrad/optimize.py` — gradient ascent: P(bind), MPNN-regularized, or on-minus-off selectivity
- `scripts/` — the gradient check and the optimization run

Built on joltz (with the affinity head) + mosaic. Needs a GPU and the Boltz-2
checkpoints.
