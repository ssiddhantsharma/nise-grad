# Levers and their sources

Design rationale for each term in the optimizer, with the primary-source formula and the verified
held-out result. Numbers are held-out Protenix interface pTM (ipTM) and gpde, best composition-passing
(maxAA < 0.35), fact-checked against the scored logs in `scripts/data/`.

## Anchors (held-out positive controls)

Provenance-documented, crystal-validated de-novo binders from Lee, Pellock, Norn et al., "Small-molecule
binding and sensing with a designed protein family", Nat Commun 2026 (doi:10.1038/s41467-026-70953-8),
defined in `scripts/data/anchors.json`:

- apx1049 (apixaban) - PDB 8VEZ / 8VFQ, ligand GG2, 119 aa. Verified held-out ipTM 0.97, gpde 0.34.
- hcy129_mpnn5 (cortisol) - PDB 8UQF, ligand HCY, 135 aa. Verified held-out ipTM 0.86, gpde 0.64.

Scramble and random sequences score 0.24-0.42, so the judge separates real binders from nonsense. Ligand
SMILES are the PDB component canonical strings, verified by InChIKey (GG2 QNZCBYKSOIHPEH; HCY
JYGXADMDTFJGBT-VWUMJDOOSA-N, full steroid stereochemistry). Cortisol (a steroid) is chemically distinct
from apixaban, a second held-out target. These binders come from NTF2-family backbone hallucination +
ProteinMPNN, the pocket/backbone-design paradigm this study argues sequence-only gradient cannot reach.

## Objectives

### Affinity head (default, `-P(bind)`)
Boltz-2 binary affinity logit, a max over interface tokens. Maximizing it reward-hacks: sparse gradients,
collapse to low-complexity sequences that fool the head. The failure the other levers exist to fix.

### pTMEnergy (`--ptm-energy`)
BindEnergyCraft (Nori et al. 2025, arXiv 2505.21241), Eq. 8. pAE logits read as an energy via a
pTM-kernel-weighted LogSumExp over distance bins, averaged over inter-chain pairs:

    E_ij = -log Σ_b g(d_b) exp(ℓ_ijb)      g(d) = 1 / (1 + (d/d0)^2)
    d0   = 1.24 (N - 15)^(1/3) - 1.8       loss = mean_{i,j inter} E_ij

Dense gradients instead of the max-based head. Verified: gpde 1.71 -> 0.85 (best structure lever), but
ipTM stays 0.44. `optimize.ptm_energy`.

## Interface / contact terms

### Contact (`--contact-weight`)
mosaic `BinderTargetContact` on the distogram, the DBMol contact loss (Qin et al. 2026, arXiv 2607.19237).
Rewards binder->target contact probability. Raises yield, does not break the ceiling.

### Confidence (`--confidence-weight`)
Mean binder->ligand interface PAE (`optimize.interface_pae`). Minor yield gain.

### Pocket-then-scaffold (`--pocket-scaffold`)
Concentrate contact on the top-12 pocket residues (`BinderTargetContact(paratope_size=12)`) and fold the
rest (`1 - mean pLDDT`), instead of one contact pressure spread over all positions.

## Estimator

### Decoupled straight-through (`--ste-backward-temp`)
Decoupled STE (arXiv 2410.13331): separate forward and backward temperatures to avoid the gradient bias
of a single-temperature STE. Our forward is a hard argmax, so the knob is the backward temperature:

    forward:  hard = one_hot(argmax(logits))
    backward: gradient flows through softmax(logits / tau_b)

Verified sweep (held-out, n=5): ipTM 0.42 / 0.42 / 0.42 at tau_b 0.25 / 0.5 / 0.75 and 0.36 at 2.0, all
far below the real binder (0.97). Decoupling the temperature does not move the ceiling, so the plateau is
not a gradient-estimator artifact.

### Anti-collapse penalties (`--composition-weight`, `--repeat-weight`)
Two differentiable simplex terms against the homopolymer collapse ("Controlling Repetition in Protein
Language Models", arXiv 2602.00782, names the symptom):

    composition_kl(soft) = KL(mean_positions(soft) || SwissProt background)   # penalize
    repetition(soft)     = mean_i <soft_i, soft_{i+1}>                        # penalize

composition_kl targets the natural amino-acid background (not plain max-entropy, itself unnatural);
repetition catches adjacent (k=2) homopolymers only. Verified: composition 0.55, repetition 0.37 held-out,
both inside the 0.36-0.55 lever band. Optimized STE designs are composition-biased (maxAA 0.21-0.24 vs a
real binder's 0.12), i.e. bias, not extreme homopolymer. `optimize.composition_kl` / `optimize.repetition`.

## Projection and the rescue

### jligandmpnn projector (`scripts/project.py`)
Fold, freeze the structure, gradient-descend the LigandMPNN NLL of the soft sequence given that structure
(`score_soft`). Verified: HURTS the held-out (0.44 -> 0.39) because it projects onto the already
reward-hacked structure. A documented negative and the control for the rescue.

### Rescue: design on a real backbone (`scripts/rescue_backbone.py`)
Same mechanism, but freeze a REAL pocket backbone. Fold apx1049 (recycling=3), freeze it, gradient-descend
the LigandMPNN NLL of a fresh random-init sequence to fit it, refold on Protenix. Verified: held-out ipTM
0.83 (n=8, up to 0.91), gpde 0.66, realistic composition (maxAA ~0.17), nearly double de-novo STE (0.45).
The only change from the failing projector is a real backbone, so the bottleneck is the backbone.
`figures/rescue.png`.

## Matched-budget baseline (`scripts/panel_baseline.py`)
Does the gradient beat gradient-free search? At a matched fold budget of 25, over the 14-ligand panel
(held-out, best composition-passing per ligand): STE 0.66 > Best-K-of-N 0.51 > O3 latent BO 0.48. The
gradient carries real signal, yet all three plateau far below the real binders (0.93): the ceiling is
shared across optimizers, so the optimizer is not the bottleneck.

## RFdiffusion3 backbone arm (`scripts/run2_rfd3_arm.sh`)
The constructive control. Generate 40 all-atom RFdiffusion3 backbones that bury apixaban, inverse-fold
each with LigandMPNN (4 sequences), score on the held-out judge. Verified: mean ipTM 0.75 (0.86 for the
best sequence per backbone), approaching the real-backbone rescue (0.83) and far above the sequence-only
plateau (0.45). A backbone generator supplies the pocket the sequence gradient cannot build.

## Not implemented (candidate next paradigm)

### HalluDesign
bioRxiv 2025.11.08.686881. Forward-pass, fine-tune-free sequence-structure co-optimization: alternate an
AlphaFold3-style diffusion module that updates the STRUCTURE (partial-diffusion / SDEdit) with an
inverse-folding model that redesigns the sequence. The RFd3 arm above already shows a backbone generator
closes most of the gap; HalluDesign would fold structure generation and sequence design into one
training-free loop, buildable from Boltz-2's diffusion module + jligandmpnn.

## Framing sources
- NISE (Polizzi lab, Nature s41586-026-10670-w), the gradient-free selection loop this is a gradient
  counterpart to.
- DRaFT (Clark et al. 2023), reward backprop through a differentiable generative model.
- O3 / oracle budgets (Kalisz et al. 2026), the matched-budget comparison run above.
