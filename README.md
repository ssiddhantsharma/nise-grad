# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Gradient design of small-molecule-binding proteins. The binder sequence is optimized directly
through a differentiable Boltz-2 affinity oracle, and every design is screened on an independent
held-out model (Protenix-2). A gradient counterpart to NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w), which does the same job by gradient-free
selection. Not a fork.

This is differentiable guidance: backprop the oracle gradient into the sequence (cf. DRaFT,
Clark et al. 2023). The rest of the guidance literature keeps the oracle black-box (FK-steering,
DPO, O3; Kalisz et al. 2026) because real biological oracles are non-differentiable. Ours is
differentiable only because the oracle is a model, which is why it is cheap and also why it
overfits.

**Status: active.** A straight-through estimator closes the soft-to-discrete gap, so STE
optimizes the real discrete design's P(bind), not a soft illusion. But a fully optimized design
overfits Boltz (poly-Q sequences reach P(bind) ~0.9 while being nonsense), so the optimized
oracle cannot be trusted. The fix is to filter on Protenix-2, validated by anchors: the real
apixaban binder scores iptm 0.98 / gpde 0.30, while scrambled and random sequences sit at ~0.25 /
gpde ~2.5. Generate-and-filter surfaces candidates approaching the real binder (best iptm ~0.86),
but the held-out ceiling is robust: contact and confidence losses and scaffold initialization all
raise yield, none break ~0.86. This mirrors DBMol (Qin et al. 2026) on the molecule side, and
matches why the field's binders come from backbone design (RFdiffusion3, SeedProteo) rather than
sequence-only gradient. Direction: Boltz-optimize, then Protenix-filter; wet lab is the real bar.

## Method
A binder sequence over the 20-residue simplex is optimized by gradient ascent through a
differentiable Boltz-2 oracle, then screened on an independent held-out model (Protenix-2).

**Objective.** For a soft sequence `m` folded with the fixed ligand, minimize

```
L(m) = -P_bind(m)  +  w_c · L_contact(m)  +  w_pae · L_conf(m)              (1)

  P_bind     Boltz-2 affinity binder/non-binder logit                  maximize    (2)
  L_contact  mosaic BinderTargetContact on the Boltz distogram:
             -log P(binder residue to ligand atom < 8 A), top-3/residue  interface (3)
  L_conf     mean binder/ligand interface PAE                            confident (4)
```

**Algorithm** (straight-through gradient design)

```
Input:  ligand, binder length N, budget K, weights (w_c, w_pae), optional scaffold s
Init:   x <- bias·one_hot(s) + noise   if scaffold   else   noise         # logits [N,20]
for k = 1 .. K:
    soft <- softmax(x)
    hard <- one_hot(argmax(soft))
    seq  <- soft + stop_grad(hard - soft)              # forward=hard, backward=soft
    pbind, out <- Boltz2(seq, ligand; recycling=3)     # fold the discrete design
    L    <- -pbind + w_c·contact(out) + w_pae·conf(out)          # eq. (1)
    x    <- Adam(x, dL/dx)
return argmax(x)                                       # a discrete binder sequence
```

**Pipeline**

```
init (random | pocket scaffold) -> STE-optimize L -> [project: LigandMPNN] -> Protenix-2 filter
```

`scripts/matched_budget.py` runs the optimizer (STE / Best-K-of-N / O3; `--contact-weight`,
`--confidence-weight`, `--init-seq`). `scripts/protenix_score.py` is the held-out filter.

## The straight-through estimator
The argmax that turns a soft distribution into a real sequence is not differentiable, so you
cannot backprop through "pick the top amino acid". Optimizing the soft sequence instead maximizes
a fiction: its optimum sits between real amino acids, so the soft P(bind) climbs to ~0.8 but the
discrete argmax, refolded, scores 0.1 to 0.3 and is degenerate (poly-Leu / poly-Phe). STE fixes
this by scoring the discrete sequence in the forward pass while letting the gradient flow through
the soft distribution:

```python
hard = one_hot(argmax(soft))
seq  = soft + stop_gradient(hard - soft)   # forward = hard, backward = soft
```

The value being maximized is now the discrete design's P(bind), while the logits still get a
gradient.

![gradient reality](figures/gradient_reality.png)

*30aa binder, recycling=3, num_sampling_steps=25, 3 seeds each. Naive soft P(bind) reaches ~0.9
but the discrete design scores 0.33 to 0.41 and is degenerate. STE optimizes the discrete design
directly: 0.44 to 0.73 with realistic composition, well-folded (119/119 backbone bonds, pLDDT
~0.8).* `scripts/optimize_ste.py`.

## The held-out judge, and a robust ceiling
STE closes the soft-to-discrete gap within Boltz, but it does not by itself make real binders. A
model optimized against its own head overfits by construction, so trusting the optimized oracle
is a mistake. Refold every design on an independent model, Protenix-2, and keep only what
survives. The metric matters: generic ipTM is too compressed to filter on (random sequences
score ~0.7), while Protenix `gpde` and `ranking_score` separate real from nonsense (anchors: real
0.98 / gpde 0.30 vs scramble/random 0.25 / gpde 2.5). As the optimization budget rises the Boltz
proxy climbs but the held-out score stalls and sequences degenerate, the same picture DBMol found
on the molecule side.

![plateau](figures/plateau.png)

Every objective lever raises held-out yield but none break the ceiling. Confidence (interface
PAE) and contact (mosaic `BinderTargetContact` on the Boltz distogram) both lift the mean, and
scaffold initialization from a real pocket fold gives the best foldedness, yet the best design
across all of them stays 0.69 to 0.87 ipTM, well below the real binder at 0.98.

![levers](figures/levers.png)

The reading: sequence-only gradient design plateaus because it optimizes the sequence but never
designs the pocket, and the field's small-molecule binders come from backbone design instead
(RFdiffusion3, the Baker NTF2 family). nise-grad is the differentiable refinement and scoring
layer, not the pocket generator.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)`, the in-loop optimizer
- `src/nisegrad/optimize.py` STE gradient ascent (optional interface-PAE confidence term)
- `scripts/matched_budget.py` generate designs (STE / Best-K-of-N / O3) at a fold budget
- `scripts/protenix_score.py` held-out Protenix-2 judge (iptm, ligand ipTM, gpde, ranking)
- `scripts/*_figure.py` the figures; `scripts/data/` the scored experiment logs
- `scripts/optimize_ste.py` a single STE run

## Install
```
pip install -e .
# GPU jax that works with mosaic/joltz on CUDA 12 (newer nvidia libs break cuSPARSE):
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```
Checkpoints in `~/.boltz` (or set `NISEGRAD_BOLTZ_CACHE`): `boltz2_conf.ckpt`, `boltz2_aff.ckpt`.
Independent designs are embarrassingly parallel, one per GPU, for campaign throughput.
