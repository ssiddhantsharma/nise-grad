# nise-grad

[![CI](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/nise-grad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Gradient design of small-molecule-binding proteins. The binder sequence is optimized directly
through a differentiable Boltz-2 affinity oracle, and every design is screened on a second,
architecturally-related model (Protenix-2). Boltz-2 and Protenix-2 are both AF3-style co-folders
trained on largely the same PDB/BioLiP data, so the judge is a held-out model but not an
independent one: correlated failure modes are expected, and a plateau seen through it may be a
shared-bias ceiling rather than a fundamental one. A gradient counterpart to NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w), which does the same job by gradient-free
selection. Not a fork.

This is differentiable guidance: backprop the oracle gradient into the sequence (cf. DRaFT,
Clark et al. 2023). The rest of the guidance literature keeps the oracle black-box (FK-steering,
DPO, O3; Kalisz et al. 2026) because real biological oracles are non-differentiable. Ours is
differentiable only because the oracle is a model, which is why it is cheap and also why it
overfits.

**Status: active.** Optimizing the *soft* sequence maximizes a fiction: the naive soft optimum
sits between amino acids and its argmax refolds to degenerate poly-X at P(bind) ~0.9. A
straight-through estimator closes that soft-to-discrete gap, so STE optimizes the real discrete
design's P(bind); the STE designs have realistic (if composition-biased) sequences, not poly-X. But
STE still overfits Boltz, so the optimized oracle cannot be trusted on its own. The fix is to filter
on Protenix-2: a real apixaban binder scores iptm 0.98 / gpde 0.30, while scrambled and random
sequences sit at ~0.25 / gpde ~2.5. Generate-and-filter surfaces a few candidates toward the real
binder (best iptm 0.86, but only 5 of 90 designs clear 0.7), and no objective lever breaks that
band. This mirrors DBMol (Qin et al. 2026) on the molecule side. All held-out results here are a
single ligand (apixaban) with single-sequence anchors, so read the ceiling as suggestive, not
established (see Limitations). Direction: Boltz-optimize, then Protenix-filter; wet lab is the real
bar.

## Method
A binder sequence over the 20-residue simplex is optimized by gradient ascent through a
differentiable Boltz-2 oracle, then screened on a second, architecturally-related model
(Protenix-2; held-out weights, not an independent oracle).

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

## The held-out judge, and the ceiling
STE closes the soft-to-discrete gap within Boltz, but it does not by itself make real binders. A
model optimized against its own head overfits by construction, so trusting the optimized oracle
is a mistake. Refold every design on a second model, Protenix-2, and keep only what survives (its
weights are held out from the optimization, though it shares Boltz's architecture family and
training data, so treat surviving as necessary, not sufficient). The metric matters: generic ipTM
is too compressed to filter on (random sequences score ~0.7), while Protenix `gpde` and
`ranking_score` separate this real binder from nonsense (anchors: real 0.98 / gpde 0.30 vs
scramble/random 0.25 / gpde 2.5, n=1 each). As the optimization budget rises the Boltz proxy climbs
while the held-out score does not follow it up (budget 25/100/250 -> 0.45/0.38/0.53, n=80/5/5), the
same picture DBMol found on the molecule side. The high-budget points are few, so this is a stall,
not a proven asymptote.

![plateau](figures/plateau.png)

No objective lever moves the best design out of the band below the real binder. Confidence
(interface PAE), contact (mosaic `BinderTargetContact` on the Boltz distogram), and scaffold
initialization from a real pocket fold each shift the mean by less than their spread at n=3-5, so
the per-lever differences are not statistically distinguishable here; what is consistent is that the
best design across all of them stays 0.69 to 0.87 ipTM, below the real binder at 0.98.

![levers](figures/levers.png)

A late projection onto the foldable manifold (DBMol's optimize-then-project idea) does not
rescue it either. Using LigandMPNN `score_soft` to gradient-descend a design toward a sequence
that folds to its own frozen structure makes the held-out score worse (0.51 to 0.39), because it
projects onto the reward-hacked structure rather than escaping it. A proper projection needs a
sampler onto the data manifold (ADFLIP, DeFoG), not a gradient to a frozen structure.
`scripts/project.py`.

The reading: sequence-only gradient design plateaus because it optimizes the sequence but never
designs the pocket, and the field's small-molecule binders come from backbone design instead.
nise-grad is the differentiable refinement and scoring layer, not the pocket generator.

## Limitations
Read the ceiling as suggestive, not established. Specifically:
- **One target.** Every held-out result is a single ligand (apixaban). The plateau is an
  n=1-target statement until it is reproduced on several diverse ligands.
- **Related, not independent, judge.** Boltz-2 and Protenix-2 share architecture family and
  training data; a design that exploits a shared bias can pass both. The ceiling may be a
  shared-bias artifact.
- **Single-sequence anchors.** The real/scramble/random calibration is one sequence each, and the
  real binder's provenance (an experimentally-validated apixaban binder at this pose?) is not yet
  documented. The gpde threshold needs several known small-molecule-binder positives.
- **Under-powered at high budget.** The held-out plateau is n=80 at budget 25 but only n=5 at
  budgets 100 and 250, and non-monotone; the informative points are few.
- **Budget is steps, not FLOPs.** An STE step is a forward + backward (~3x a fold); Best-K-of-N and
  O3 spend one forward fold per unit budget, so equal "budget" is not equal compute.
- **In silico only.** No wet-lab validation; all claims are model-internal.

## Layout
- `src/nisegrad/oracle.py` differentiable `P(bind)(sequence, ligand)`, the in-loop optimizer
- `src/nisegrad/optimize.py` STE gradient ascent (optional interface-PAE confidence term)
- `scripts/matched_budget.py` generate designs (STE / Best-K-of-N / O3) at a fold budget
- `scripts/protenix_score.py` held-out Protenix-2 judge (iptm, ligand ipTM, gpde, ranking)
- `scripts/project.py` late-projection onto the foldable manifold (LigandMPNN `score_soft`,
  via `src/nisegrad/boltz_ligand.py` + `ligand_mpnn_reg.py`)
- `scripts/*_figure.py` the figures; `scripts/data/` the scored experiment logs
- `scripts/optimize_ste.py` a single STE run

## References and credit
nise-grad borrows from and builds on:
- NISE (Polizzi lab), the gradient-free selection loop this is a gradient counterpart to.
  https://www.nature.com/articles/s41586-026-10670-w
- DRaFT (Clark et al. 2023), reward backprop through a differentiable generative model.
- O3 / oracle budgets (Kalisz et al. 2026), the black-box guidance literature we position against.
- DBMol (Qin et al. 2026), the molecule-side mirror; corroborates the plateau and reward-hacking
  and supplies the contact-loss idea. https://arxiv.org/abs/2607.19237
- BindEnergyCraft (Nori et al. 2025), pTMEnergy, the energy objective that resists reward-hacking
  (`--ptm-energy`). https://arxiv.org/abs/2505.21241
- L-Caliby / Caliby (Shuai et al.), the pocket-then-scaffold decomposition (`--pocket-scaffold`).
  https://github.com/ProteinDesignLab/caliby

Tools: Boltz-2 (via joltz), Protenix-2, mosaic (`BinderTargetContact` and losses), and LigandMPNN
(Dauparas et al.) via [jligandmpnn](https://github.com/ssiddhantsharma/jligandmpnn).

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
