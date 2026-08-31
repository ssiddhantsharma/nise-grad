#!/usr/bin/env bash
# RFd3 backbone arm over a slice of the 14-ligand panel on ONE GPU.
# RFd3 backbones (bury ligand) -> LigandMPNN seqs -> held-out Protenix. Resumable (skips done ligands).
# Usage: CHEM_FAP_DIR=... PROTENIX_DIR=... GPU=0 LIG_START=0 LIG_END=7 [ND=24 NS=4] bash panel_rfd3.sh
set -uo pipefail
GPU="${GPU:-0}"; LIG_START="${LIG_START:-0}"; LIG_END="${LIG_END:-14}"
ND="${ND:-24}"; NS="${NS:-4}"
CF="${CHEM_FAP_DIR:?set CHEM_FAP_DIR to your RFd3 / chemistry_fap pipeline checkout}"
NV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$NV/.venv/bin/python"
PANEL="$NV/scripts/data/ligand_panel.json"
OUTDIR="$NV/scripts/data/rfd3_panel"; mkdir -p "$OUTDIR"

mapfile -t LINES < <("$PY" -c "
import json, re
d = json.load(open('$PANEL'))
for i in range($LIG_START, min($LIG_END, len(d))):
    p = d[i]
    slug = re.sub(r'[^a-z0-9]+','_',p['ligand_name'].lower()).strip('_')[:16]
    print(f\"{i}\t{slug}\t{p['ligand']}\t{len(p['anchor_seq'])}\")
")

for line in "${LINES[@]}"; do
  i=$(cut -f1 <<<"$line"); slug=$(cut -f2 <<<"$line"); smi=$(cut -f3 <<<"$line"); L=$(cut -f4 <<<"$line")
  out="$OUTDIR/${slug}.json"
  [ -f "$out" ] && { echo "[skip] $slug done"; continue; }
  rescode=$(printf "L%02d" "$i")
  echo "===== [$i] $slug (len $L, $rescode) GPU $GPU ND=$ND ====="
  cd "$CF"
  export DYE="$slug" DYE_SMILES="$smi" LIG_RESNAME="$rescode" RUN_NAME="rfd3_$slug" \
         BINDER_LEN="$L" N_DESIGNS="$ND" N_SEQS="$NS" ALLOWED_GPUS="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
         FOLD_CLOSED=0
  source paths.sh
  PREP=$(python scripts/ligand_prep.py) || { echo "PREP FAIL $slug"; continue; }
  echo "$PREP"
  # steer H-bonds to this ligand's actual oxygens (up to 3); 'none' = burial only (no oxygens)
  OX=$(echo "$PREP" | grep -oE "'O[0-9]+'" | tr -d "'" | head -3 | paste -sd, -)
  export HBOND_LIGAND_ATOM="${OX:-none}"
  echo "hbond acceptors: $HBOND_LIGAND_ATOM"
  bash scripts/rfd3.sh          || { echo "RFD3 FAIL $slug"; continue; }
  bash scripts/ligandmpnn.sh    || { echo "MPNN FAIL $slug"; continue; }
  RUN="$CF/runs/rfd3_$slug"
  J="$OUTDIR/${slug}_designs.json"
  python3 - "$RUN/ligandmpnn/seqs" "$smi" "$J" <<'PY'
import sys, json, glob, os
seqdir, smi, out = sys.argv[1:4]; rows = []
for fa in sorted(glob.glob(os.path.join(seqdir, "*.fa"))):
    lines = [l.strip() for l in open(fa) if l.strip()]; bb = os.path.basename(fa)[:-3]; i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            hdr, seq = lines[i], (lines[i + 1] if i + 1 < len(lines) else "")
            if "id=" in hdr and seq and not seq.startswith(">"):
                rows.append({"method": "rfd3_mpnn", "seed": int(hdr.split("id=")[1].split(",")[0]),
                             "budget": 0, "ligand": smi, "seq": seq, "note": bb})
            i += 2
        else: i += 1
json.dump(rows, open(out, "w"), indent=2); print("designs", len(rows))
PY
  cd "$NV"
  export PROTENIX_DIR="${PROTENIX_DIR:?set PROTENIX_DIR to your Protenix checkout}"; O="$OUTDIR/${slug}_out"; mkdir -p "$O"
  "$PY" scripts/protenix_score.py build --in "$J" --input-json "$O/in.json"
  ( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/protenix pred -i "$O/in.json" \
      -o "$O/pred" -s 101 -n protenix-v2 --use_msa false --use_default_params true )
  "$PY" scripts/protenix_score.py parse --in "$J" --outdir "$O/pred" --seed 101
  cp "$J" "$out"
  echo "LIGAND_DONE $slug"
done
echo "PANEL_SLICE_DONE gpu=$GPU $LIG_START-$LIG_END"
