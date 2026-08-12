# nise-grad

Gradient-based, differentiable design of small-molecule–binding proteins.

This is an independent reimplementation of the ideas in **NISE** (Neural Iterative
Selection-Expansion; Polizzi lab) with one change: instead of a gradient-free
selection–expansion loop that shells out to structure predictors, `nise-grad`
optimizes the binder sequence **by gradient descent** through a differentiable
structure model, using a differentiable P(bind).

- **Differentiable oracle:** [`joltz`](https://github.com/nboyd/joltz) (a JAX port
  of Boltz-2) with its binding-affinity head, plus
  [`mosaic`](https://github.com/escalante-bio/mosaic) for the optimizer and losses.
- **Objectives:** P(bind) / structural confidence, and a paired **selectivity** loss
  (on-target minus off-target), which is a single subtraction in the gradient setting.

Credit: the method and problem framing come from NISE
([paper](https://www.nature.com/articles/s41586-026-10670-w)); this repo is a
from-scratch gradient variant, not a fork of that code.

## Status

Early / spike stage. First milestone is a feasibility check that the P(bind)
gradient flows to the sequence with a ligand present (`scripts/spike_affinity_gradient.py`).
