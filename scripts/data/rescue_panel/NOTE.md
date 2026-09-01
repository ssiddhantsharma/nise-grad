# rescue_panel: the failed gradient-descent rescue run (kept as a documented negative)

Raw output of `panel_rescue.sh` at `--comp-weight 0.0` (60-step gradient descent of the LigandMPNN NLL on
each real anchor backbone, 8 seeds). Under the paper's composition guard (maxAA < 0.35) it **collapsed**:
only 4/14 ligands (apixaban, dfhbi, dfhbi_1t, cortisol) produced valid designs; the other 10/14
reward-hacked to poly-alanine (maxAA 0.36-0.61), gaming interface pTM the same way the affinity gradient
does. The optimiser reward-hacks the *inverse-folding* score just like it reward-hacks the affinity head.

This backs no number in the paper -- the paper reports the apixaban rescue from `rescue_scored.json` only.
An earlier commit message here claimed "0.91, near real, above RFd3"; that was wrong (it ignored the guard).

**Fix:** `rescue_backbone.py` now takes `--comp-weight` (default 1.0 in `panel_rescue.sh`). Re-run for a
composition-passing panel rescue. This directory is kept as the "before" of that fix.
