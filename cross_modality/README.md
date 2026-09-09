# Cross-modality check — the oracle findings on de novo *protein* binders

The paper's small-molecule findings (the co-folding oracle places binders but does not rank
affinity; maximising it reward-hacks) are tested here on an independent, larger, different-modality
set: **1,320 de novo protein binders** (354 with a measured K_d, 13 targets, 10 co-folding models)
from `Anthropic/claude-protein-binder-design`. Pure CPU — the folding is already done in the release.

## Run

```bash
bash fetch.sh                       # ~10 MB, two parquets to /tmp (no GPU, no auth)
python make_metric_analysis.py      # -> metric_analysis.csv  (per-metric AUROC + within-target K_d rho)
python make_per_model.py            # -> per_model.csv         (per-model + ensemble: combining test)
python make_headline.py             # -> plots_anthropic/metric_headline.png
python make_reward_hacking.py       # -> plots_anthropic/reward_hacking.png + reward_hacking.csv
python make_trivial_baseline.py     # -> plots_anthropic/trivial_baseline.png + trivial_baseline.csv
```
Needs `pandas numpy scipy scikit-learn matplotlib pyarrow`. `tmol_ddg_kd.csv` (physics ΔG_bind,
joined for the K_d-wall comparison) ships here.

## What it shows

- **Placement, not affinity.** Interface-confidence metrics separate binders from non-binders at
  AUROC ≈ 0.77 (ipSAE, interface PAE); no metric ranks K_d among binders above ρ ≈ +0.27 (the K_d wall).
- **Combining doesn't rescue.** The 10-model ensemble beats neither the best single metric (AP) nor the
  best single model (K_d). Matches Anthropic's own reported 0.66 AP / 0.25 within-target K_d ρ.
- **Reward-hacking signature** (`make_reward_hacking.py`). 18% of confirmed non-binders outscore the
  median real binder on consensus interface pTM; selecting the top 10% by the oracle's own score yields
  only 58% binders (base 27%), the top 5% less pure still — the protein echo of the poly-alanine hack.
- **A "cLogP for proteins"** (`make_trivial_baseline.py`). Binder length alone reaches within-target
  K_d ρ = +0.18 vs the best co-folding metric's +0.27 — the oracle keeps only a modest, non-significant
  edge over triviality.

These are the appendix cross-modality figures in the paper. The design-mechanism findings (gradient
plateaus, backbone is the limit) do **not** come from this data — the Anthropic pipeline is
generate-and-filter, not gradient hallucination, so it carries no optimisation trajectories.
