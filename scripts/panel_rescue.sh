#!/usr/bin/env bash
# Panel-wide real-backbone rescue. For each anchor in the 14-ligand panel: freeze the real pocket
# backbone and design NSEEDS sequences on it, then score them on the held-out Protenix judge. One
# process per seed on purpose -- rescue_backbone.py leaks a JAX tracer on a second jit in-process.
# Resumable: skips a ligand whose scored file already exists.
#
# Usage:
#   PROTENIX_DIR=... LIGANDMPNN_CKPT=... \
#   GPU=1 NSEEDS=4 bash scripts/panel_rescue.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
PANEL="$REPO/scripts/data/ligand_panel.json"
OUTDIR="$REPO/scripts/data/rescue_panel"; mkdir -p "$OUTDIR"
GPU="${GPU:-1}"; NSEEDS="${NSEEDS:-4}"; LIG_START="${LIG_START:-0}"; LIG_END="${LIG_END:-14}"
# COMP_WEIGHT>0 penalises low-complexity sequences. The first panel run at 0.0 collapsed to
# poly-alanine on 10/14 ligands (MPNN-NLL descent reward-hacks), so the default is now 1.0.
COMP_WEIGHT="${COMP_WEIGHT:-1.0}"
: "${PROTENIX_DIR:?set PROTENIX_DIR to your Protenix checkout}"
: "${LIGANDMPNN_CKPT:?set LIGANDMPNN_CKPT}"

mapfile -t LINES < <("$PY" -c "
import json, re
d = json.load(open('$PANEL'))
for i in range($LIG_START, min($LIG_END, len(d))):
    p = d[i]
    slug = re.sub(r'[^a-z0-9]+', '_', p['ligand_name'].lower()).strip('_')[:20]
    print(f\"{slug}\t{p['ligand']}\t{p['anchor_seq']}\")
")

for line in "${LINES[@]}"; do
  slug=$(cut -f1 <<<"$line"); smi=$(cut -f2 <<<"$line"); seq=$(cut -f3 <<<"$line")
  scored="$OUTDIR/${slug}.json"
  [ -f "$scored" ] && { echo "[skip] $slug already scored"; continue; }
  J="$OUTDIR/${slug}_designs.json"; rm -f "$J"
  echo "===== $slug (L=${#seq}) GPU $GPU NSEEDS=$NSEEDS ====="
  for s in $(seq 0 $((NSEEDS - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/rescue_backbone.py" \
      --ref-seq "$seq" --ligand "$smi" --seed "$s" --steps 60 --comp-weight "$COMP_WEIGHT" --out "$J" \
      || echo "RESCUE FAIL $slug seed $s"
  done
  [ -f "$J" ] || { echo "no designs for $slug"; continue; }
  O="$OUTDIR/${slug}_out"; mkdir -p "$O"
  "$PY" "$REPO/scripts/protenix_score.py" build --in "$J" --input-json "$O/in.json"
  ( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/protenix pred -i "$O/in.json" \
      -o "$O/pred" -s 101 -n protenix-v2 --use_msa false --use_default_params true )
  "$PY" "$REPO/scripts/protenix_score.py" parse --in "$J" --outdir "$O/pred" --seed 101
  cp "$J" "$scored"
  echo "LIGAND_DONE $slug"
done
echo "PANEL_RESCUE_DONE gpu=$GPU nseeds=$NSEEDS $LIG_START-$LIG_END"
