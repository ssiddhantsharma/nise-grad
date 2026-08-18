# Levers and their sources

Design rationale for each term in the optimizer, with the primary-source formula and a verbatim
quote where the claim is load-bearing. Verified against the source PDF/HTML, not a summary.

## Benchmark anchors and provenance

The held-out filter is only as good as its positive control. The original `anchor_real` sequence
(`APSLEEQ...`, 123 aa) has no documented provenance and, on inspection, does not match the deposited
crystal-structure de-novo apixaban binder; its 0.98 score cannot be relied on. It is retired in
favour of provenance-documented, crystal-validated anchors from Lee, Pellock, Norn et al.,
"Small-molecule binding and sensing with a designed protein family", Nat Commun 2026
(doi:10.1038/s41467-026-70953-8). These are defined in `scripts/data/anchors.json`:

- apx1049 (apixaban) - PDB 8VEZ (2.15 A) / 8VFQ (2.10 A), ligand GG2, 119 aa.
- hcy129_mpnn5 (cortisol) - PDB 8UQF (1.52 A), ligand HCY, 135 aa.

Ligand SMILES are the PDB chemical-component canonical strings, verified by InChIKey (GG2 apixaban
QNZCBYKSOIHPEH, achiral; HCY cortisol JYGXADMDTFJGBT-VWUMJDOOSA-N, full steroid stereochemistry -
the non-stereo string silently loses it). The repo's existing apixaban SMILES is confirmed identical
to GG2 (same InChIKey), so the ligand was always correct; only the anchor protein was wrong. These
binders come from NTF2-family backbone hallucination + ProteinMPNN, i.e. pocket/backbone design -
the paradigm this repo's characterization argues sequence-only gradient cannot reach, so they double
as citable corroboration. Cortisol (a steroid) is chemically distinct from apixaban, giving a second
target for the held-out ceiling. Establishing anchor scores and the cortisol held-out needs folds
(GPU); the definitions and verification here do not.

## Objectives

### Affinity head (default, `-P(bind)`)
Boltz-2 binary affinity logit. Maximizing it reward-hacks: the head is a max over interface
tokens, so gradients are sparse and the optimizer collapses to low-complexity sequences that fool
the head. This is the failure the other levers exist to fix.

### pTMEnergy (`--ptm-energy`)
BindEnergyCraft (Nori et al. 2025, arXiv 2505.21241), Eq. 8. The pAE logits are read as an
energy via a pTM-kernel-weighted LogSumExp over distance bins, averaged over inter-chain pairs:

    E_ij = -log Σ_b g(d_b) exp(ℓ_ijb)          g(d) = 1 / (1 + (d/d0)^2)
    d0   = 1.24 (N - 15)^(1/3) - 1.8           loss = mean_{i,j inter} E_ij

Dense gradients (a soft sum over bins) instead of the max-based affinity head. Verified result in
our runs: gpde 1.71 -> 0.85 (best structure lever), no poly-Q, but ipTM stays ~0.44. Implemented in
`optimize.ptm_energy`.

## Interface / contact terms

### Contact (`--contact-weight`)
mosaic `BinderTargetContact` on the distogram, the DBMol contact loss (Qin et al. 2026,
arXiv 2607.19237, Eq. 3-6). Rewards binder->target contact probability. Raises yield, does not
break the ceiling.

### Confidence (`--confidence-weight`)
Mean binder->ligand interface PAE (`optimize.interface_pae`). Minor yield gain.

### Pocket-then-scaffold (`--pocket-scaffold`)
L-Caliby / Caliby (Shuai et al., github.com/ProteinDesignLab/caliby). Concentrate contact on the
top-12 pocket residues (`BinderTargetContact(paratope_size=12)`) and fold the rest
(`1 - mean pLDDT`), instead of one contact pressure spread over all positions.

## Estimator

### Decoupled straight-through (`--ste-backward-temp`)
Decoupled STE (arXiv 2410.13331). Standard STE couples forward and backward through one
temperature; the paper shows this causes "gradient bias" and convergence to "degenerate solutions".
Fix: separate temperatures. Our forward is already a hard argmax, so the tunable knob is the
backward temperature:

    forward:  hard = one_hot(argmax(logits))
    backward: gradient flows through softmax(logits / tau_b)
    seq = softmax(logits/tau_b) + stop_gradient(hard - softmax(logits/tau_b))

tau_b < 1 sharpens the gradient (exploitation), tau_b = 1 is the plain STE (byte-identical).
CPU-verified: forward stays hard at every tau_b; grad-norm 19.5 / 7.9 / 3.6 at tau_b 0.5 / 1.0 / 2.0.
Note: this applies the decoupling principle to a hard-forward STE; it is not the paper's exact
soft-forward variant.

### Anti-collapse penalties (`--composition-weight`, `--repeat-weight`)
"Controlling Repetition in Protein Language Models" (arXiv 2602.00782). Names our exact symptom:
"A single residue is extended into long runs (e.g., AAAAAA) ... a global collapse into a
low-complexity sequence dominated by one amino acid type." Their metrics:

    H_norm(x)     = -Σ_a p(a) log2 p(a) / log2|A|            (normalized token entropy, [0,1])
    Distinct-n(x) = |unique n-grams| / |n-grams|             (n = 2,3)
    R_hpoly(x)    = 1 - (1/T) Σ_i ℓ_i · 1(ℓ_i >= k),  k = 4  (homopolymer score)

Their method (UCCS) is inference-time activation steering on a PLM's hidden states, "without
retraining"; it does not operate on a sequence-logit simplex, so it does not port to our gradient
loop. R_hpoly itself is non-differentiable (hard run-length + indicator). We therefore hand-roll
two differentiable simplex terms:

    composition_kl(soft) = KL(mean_positions(soft) || SwissProt background)   # penalize
    repetition(soft)     = mean_i <soft_i, soft_{i+1}>                        # penalize

composition_kl targets the natural amino-acid background, not a plain max-entropy reward: max
entropy would push toward a uniform 5%-each composition, which is itself unnatural, whereas the KL
matches real frequencies. repetition catches only adjacent (k=2) repetition, i.e. R_hpoly-style
homopolymers, not periodic motifs (AGAG scores ~0); it is an anti-homopolymer term, not a general
anti-repeat one. Data-side motivation in `figures/collapse.png` (optimized designs maxAA 0.21-0.24
vs real binder 0.12, though 0.21 is biased composition, not extreme homopolymer). Implemented in
`optimize.composition_kl` / `optimize.repetition`.

## Late projection

### jligandmpnn projector (`scripts/project.py`)
Fold, freeze the structure, gradient-descend the LigandMPNN NLL of the soft sequence given that
structure (`score_soft`). Result: HURTS the held-out (0.51 -> 0.39) because it projects onto the
already-reward-hacked structure. Kept as a documented negative.

## Not implemented (verified, candidate next paradigm)

### HalluDesign
bioRxiv 2025.11.08.686881 (repo: github.com/MinchaoFang/HalluDesign). Verified from the abstract:
"iteratively update protein structure and sequence ... enables fine-tune free, forward-pass only
sequence-structure co-optimization." The loop alternates (1) an AlphaFold3-style diffusion module
that hallucinates/updates the STRUCTURE conditioned on the current sequence, at varying noise
levels (partial-diffusion / SDEdit style), and (2) an inverse-folding model (ProteinMPNN or
LigandMPNN) that redesigns the sequence for the updated structure. It is forward-pass only, not
gradient-based. It targets our root cause (sequence-only gradient never moves the pocket) and could
be built training-free from components we already have: Boltz-2's diffusion module for the
structure stage and jligandmpnn (LigandMPNN) for the sequence stage.

## Framing sources

- NISE (Polizzi lab, Nature s41586-026-10670-w), the gradient-free selection loop this is a
  gradient counterpart to.
- DRaFT (Clark et al. 2023), reward backprop through a differentiable generative model.
- O3 / oracle budgets (Kalisz et al. 2026), the matched-budget comparison.
