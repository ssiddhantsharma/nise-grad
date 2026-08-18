#!/usr/bin/env bash
# Fold the verified anchors and run the deferred held-out experiments, then score on Protenix-2.
#
# Usage (needs a free GPU; nothing here runs without an explicit invocation):
#   git pull                                   # bizon repo must have the committed lever flags
#   bash scripts/fold_driver.sh anchors        # A3: fold apx1049 + hcy129 anchors + negatives
#   bash scripts/fold_driver.sh cortisol       # A2: held-out STE vs cortisol
#   bash scripts/fold_driver.sh budget         # B1: held-out STE at budgets 25/100/250, n=20
#   bash scripts/fold_driver.sh levers         # decoupled-STE / composition / repetition
#   bash scripts/fold_driver.sh all            # all of the above, in order
# Detach for a long run:  setsid nohup bash scripts/fold_driver.sh all > /tmp/fold.log 2>&1 < /dev/null &
#
# Env: CUDA_VISIBLE_DEVICES (default 0), PROTENIX_DIR, OUT dir, CORT_LEN, N_BUDGET.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
MB="$REPO/scripts/matched_budget.py"
PS="$REPO/scripts/protenix_score.py"
ANCHORS="$REPO/scripts/data/anchors.json"
PROTENIX_DIR="${PROTENIX_DIR:-$HOME/siddhant/tools/Protenix}"
OUT="${OUT:-/tmp/nisegrad_run}"; mkdir -p "$OUT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED=101
APIX_LEN=123
CORT_LEN="${CORT_LEN:-120}"
N_BUDGET="${N_BUDGET:-20}"

# apixaban SMILES, verified identical to PDB ligand GG2 (InChIKey QNZCBYKSOIHPEH). Cortisol is
# read from anchors.json (stereo SMILES) so we never fold the wrong stereochemistry.
APIX_SMILES='COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N'
CORT_SMILES="$("$PY" -c "import json,sys;print(next(r['ligand_smiles'] for r in json.load(open('$ANCHORS')) if r['target']=='cortisol'))")"

PHASES="$*"
[ -z "$PHASES" ] && { echo "usage: bash scripts/fold_driver.sh [anchors|cortisol|budget|levers|all]"; exit 1; }
case " $PHASES " in *" all "*) PHASES="anchors cortisol budget levers";; esac
has() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "GPU=$GPU  OUT=$OUT  phases: $PHASES"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || true

# real + scramble + random anchor rows for one target, negatives deterministic (seed 0).
build_anchors() {  # $1 target  $2 out.json  $3 real-label
  "$PY" - "$ANCHORS" "$1" "$2" "$3" <<'PYEOF'
import json, sys, random
anchors, target, out, real_label = sys.argv[1:5]
AA = "ARNDCQEGHILKMFPSTWYV"
rec = next(r for r in json.load(open(anchors)) if r["target"] == target)
seq, smi = rec["seq"], rec["ligand_smiles"]
random.seed(0)
scr = list(seq); random.shuffle(scr); scr = "".join(scr)
rnd = "".join(random.choice(AA) for _ in seq)
rows = [
    {"method": real_label,       "seed": 0, "budget": 0, "ligand": smi, "seq": seq, "note": rec["name"]},
    {"method": "anchor_scramble", "seed": 0, "budget": 0, "ligand": smi, "seq": scr},
    {"method": "anchor_random",   "seed": 0, "budget": 0, "ligand": smi, "seq": rnd},
]
json.dump(rows, open(out, "w"), indent=2)
print(f"built {len(rows)} anchor rows for {target} -> {out}")
PYEOF
}

# append N STE designs to a designs JSON. Extra STE flags after the 5th arg. One process per seed
# (a second optimize/jit in one process leaks a JAX tracer).
gen_ste() {  # $1 out.json  $2 ligand  $3 binder-len  $4 checkpoints  $5 nseeds  [extra flags...]
  local J="$1" LIG="$2" LEN="$3" CP="$4" N="$5"; shift 5
  for s in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$MB" --method ste --seed "$s" \
      --checkpoints "$CP" --binder-len "$LEN" --ligand "$LIG" --out "$J" "$@"
  done
}

run_protenix() {  # $1 designs.json
  local J="$1" NAME; NAME="$(basename "$J" .json)"
  "$PY" "$PS" build --in "$J" --input-json "$OUT/${NAME}_in.json"
  ( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/protenix pred \
      -i "$OUT/${NAME}_in.json" -o "$OUT/${NAME}_out" -s "$SEED" -n protenix-v2 \
      --use_msa false --use_default_params true > "$OUT/${NAME}_pred.log" 2>&1 )
  "$PY" "$PS" parse --in "$J" --outdir "$OUT/${NAME}_out" --seed "$SEED"
}

if has anchors; then
  echo "== A3: anchor calibration (does a real crystal binder score high?) =="
  build_anchors apixaban "$OUT/apix_anchors.json" anchor_real; run_protenix "$OUT/apix_anchors.json"
  build_anchors cortisol "$OUT/cort_anchors.json" anchor_real; run_protenix "$OUT/cort_anchors.json"
fi

if has cortisol; then
  echo "== A2: held-out STE vs cortisol =="
  build_anchors cortisol "$OUT/cortisol.json" anchor_real
  gen_ste "$OUT/cortisol.json" "$CORT_SMILES" "$CORT_LEN" "25" 20
  run_protenix "$OUT/cortisol.json"
fi

if has budget; then
  echo "== B1: held-out STE at budgets 25/100/250, n=$N_BUDGET =="
  build_anchors apixaban "$OUT/budget.json" anchor_real
  gen_ste "$OUT/budget.json" "$APIX_SMILES" "$APIX_LEN" "25,100,250" "$N_BUDGET"
  run_protenix "$OUT/budget.json"
fi

if has levers; then
  echo "== levers: decoupled-STE / composition / repetition =="
  for spec in "dste:--ste-backward-temp 0.5" "comp:--composition-weight 0.1" "rep:--repeat-weight 0.1"; do
    name="${spec%%:*}"; flags="${spec#*:}"
    J="$OUT/lever_${name}.json"
    build_anchors apixaban "$J" anchor_real
    gen_ste "$J" "$APIX_SMILES" "$APIX_LEN" "25" 5 $flags
    run_protenix "$J"
  done
fi

echo "ALLDONE  results in $OUT/*.json"
