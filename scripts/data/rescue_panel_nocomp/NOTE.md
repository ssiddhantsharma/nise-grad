# rescue_panel_nocomp: the failed gradient-descent rescue run (kept as a documented negative)

Raw output of `panel_rescue.sh` at `--comp-weight 0.0` (60-step gradient descent of the LigandMPNN NLL on
each real anchor backbone, 8 seeds, **no composition penalty**). Under the paper's composition guard
(maxAA < 0.35) it **collapsed**: only 4/14 ligands (apixaban, dfhbi, dfhbi_1t, cortisol) produced valid
designs; the other 10/14 reward-hacked to poly-alanine (maxAA 0.36-0.61), gaming interface pTM the same
way the affinity gradient does. The optimiser reward-hacks the *inverse-folding* score just like the
affinity head.

**Fix and result:** `rescue_backbone.py --comp-weight` adds a `composition_kl` penalty (`panel_rescue.sh`
default 1.0). The corrected run is in `../rescue_panel/`: all 14 ligands composition-passing, per-ligand
best 0.95 (mean 0.91), near/at the real binder. This directory is kept as the "before" of that fix; it
backs no number in the paper.
