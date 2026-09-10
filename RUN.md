# RUN — coauthor quickstart (GPU)

What you can run from this repo **today**, one command each. This reproduces (and can extend) the
paper's small-molecule results on your GPU. Everything here is protein–**ligand** (the paper's scope);
protein–protein is not in this repo yet (see the bottom).

## 0. One-time setup (~15 min)

```bash
uv sync                                              # build .venv from the lockfile
uv pip install "jax[cuda12]==0.10.1" nvidia-cudnn-cu12==9.17.0.29 \
   nvidia-cusolver-cu12==11.7.3.90 nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5   # GPU JAX
bash scripts/get_weights.sh                          # LigandMPNN checkpoint (rescue/guided only)
```

**The one external dependency — the held-out judge.** Scoring runs a separate ByteDance **Protenix-v2**
checkout with its own **Python 3.11** venv (cp311 CUDA kernels won't import under 3.12), weights from the
mirror `huggingface.co/TMF001/protenix-v2-weights`. Point `PROTENIX_DIR` at it. Boltz-2 weights download
themselves on the first fold.

```bash
export PROTENIX_DIR=/path/to/Protenix          # the 3.11 judge checkout
export LIGANDMPNN_CKPT=$PWD/weights/ligandmpnn_v_32_010_25.pt
export CUDA_VISIBLE_DEVICES=0
```

## 1. Verify the box before spending GPU

```bash
bash scripts/check_setup.sh        # doctor: venv, deps import, jax-on-GPU, Boltz, LigandMPNN, Protenix judge
```
Green on the venv / Boltz / Protenix rows = you're ready. It never downloads and always exits 0.

## 2. Reproduce the paper's core (one command, detached)

```bash
setsid nohup bash scripts/fold_driver.sh all > /tmp/fold.log 2>&1 < /dev/null &
tail -f /tmp/fold.log
```
Phases (`anchors cortisol budget levers`, run in order): anchor calibration, held-out STE vs a second
target, the budget sweep (25/100/250 folds), and the estimator/composition/repetition levers. Results land
as JSON in `$OUT` (default `/tmp/nisegrad_run`). Compare them to the committed `scripts/data/*.json` behind
the paper's numbers, or regenerate figures with the `scripts/*_figure.py` scripts. Run a single phase with
`bash scripts/fold_driver.sh budget`.

**Rough cost:** the full chain is a few hundred folds on one A100/H100 — order $30–60. Start with
`bash scripts/fold_driver.sh anchors` (cheapest) to confirm a real crystal binder scores high before
launching `all`.

## 3. Easy value-add — more design seeds (tighter best-of-N + design-variance bars)

Run more STE **design** seeds per ligand, then report per-ligand mean ± std and a tighter per-ligand best:

```bash
OUT=/tmp/panel16 .venv/bin/python scripts/panel_run.py ligands --nseeds 16
```
`panel_run.py ligands` reads `scripts/data/ligand_panel.json` and folds real anchor + STE designs per
ligand, scoring on the Protenix judge unless you pass `--no-score`. Note this varies the **design** seed;
the judge fold seed is fixed (`-s 101`). The *fold*-sampling variance the benchmark caveat mentions (E7)
would instead sweep that judge seed — a one-line change in `panel_run.py:protenix` / `fold_driver.sh` — so
flag it to Sid rather than assuming `--nseeds` covers it.

## Guardrails
- **Detach long runs** (`setsid nohup … &`), or a client disconnect kills the job.
- **Fresh `$OUT` per run** — a reused dir merges JSON rows silently.
- One STE seed per process (a second JIT in one process leaks a JAX tracer — `fold_driver.sh` already loops
  one process per seed; keep that if you edit it).

## What is NOT runnable here yet
**Protein–protein binder design (handoff experiments E1/E2).** This repo is protein–ligand throughout
(`matched_budget.py --ligand <SMILES>`, a ligand-aware oracle, ligand anchors). Generalising the gradient
plateau and the backbone-rescue to protein targets needs a new protein-target oracle feature path — a code
change, not a flag — and must be GPU-validated before it's a "run it" task. Ask Sid before starting that;
it's a sprint, not a config.
