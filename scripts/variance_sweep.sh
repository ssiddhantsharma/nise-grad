#!/usr/bin/env bash
# E7 — judge fold-seed variance. Fold each panel anchor (real binder) on the
# Protenix-v2 judge at N seeds and report per-design interface-pTM mean +/- std,
# quantifying the "single seeded fold" caveat (diffusion-sampling variation).
# Re-scores FIXED sequences, so this isolates JUDGE variance (not design variance).
#
#   PROTENIX_DIR=~/Protenix CUDA_VISIBLE_DEVICES=0 bash scripts/variance_sweep.sh
#   SEEDS="101 102 103 104 105" LIGS="apixaban cortisol ..." bash scripts/variance_sweep.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
PS="$REPO/scripts/protenix_score.py"
PANEL="$REPO/scripts/data/ligand_panel.json"
PROTENIX_DIR="${PROTENIX_DIR:-$HOME/Protenix}"
OUT="${OUT:-/tmp/variance_sweep}"; mkdir -p "$OUT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEEDS="${SEEDS:-101 102 103 104 105}"

# one anchor (real binder) row per panel ligand -> designs.json
"$PY" - "$PANEL" "$OUT/designs.json" "${LIGS:-}" <<'PYEOF'
import json, sys
panel = json.load(open(sys.argv[1]))
want = set(sys.argv[3].split()) if len(sys.argv) > 3 and sys.argv[3] else None
rows = [{"method": "anchor_real", "seed": 0, "budget": 0,
         "ligand": e["ligand"], "seq": e["anchor_seq"],
         "note": e["ligand_name"], "ligand_name": e["ligand_name"]}
        for e in panel if (want is None or e["ligand_name"] in want)]
json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(f"built {len(rows)} anchor rows -> {sys.argv[2]}")
PYEOF

"$PY" "$PS" build --in "$OUT/designs.json" --input-json "$OUT/in.json"

for s in $SEEDS; do
  echo "== judge seed $s =="
  ( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/protenix pred \
      -i "$OUT/in.json" -o "$OUT/out_$s" -s "$s" -n protenix-v2 \
      --use_msa false --use_default_params true > "$OUT/pred_$s.log" 2>&1 )
  cp "$OUT/designs.json" "$OUT/scored_$s.json"
  "$PY" "$PS" parse --in "$OUT/scored_$s.json" --outdir "$OUT/out_$s" --seed "$s"
done

# aggregate: per-ligand mean +/- std of protenix_iptm across seeds
"$PY" - "$OUT" $SEEDS <<'PYEOF'
import json, sys, glob, statistics as st
out = sys.argv[1]; seeds = sys.argv[2:]
byid = {}
for s in seeds:
    for r in json.load(open(f"{out}/scored_{s}.json")):
        v = r.get("protenix_iptm")
        if v is not None:
            byid.setdefault(r["ligand_name"], []).append(float(v))
rows, stds = [], []
for lig, vs in sorted(byid.items()):
    if len(vs) >= 2:
        m, sd = st.mean(vs), st.pstdev(vs)
        rows.append({"ligand": lig, "n_seeds": len(vs), "mean_iptm": round(m, 4), "std_iptm": round(sd, 4)})
        stds.append(sd)
json.dump(rows, open(f"{out}/variance_sweep.json", "w"), indent=1)
print(f"\n{'ligand':<16}{'mean':>8}{'std':>8}{'n':>4}")
for r in rows:
    print(f"{r['ligand']:<16}{r['mean_iptm']:>8.3f}{r['std_iptm']:>8.3f}{r['n_seeds']:>4}")
if stds:
    print(f"\nmedian per-design std across {len(seeds)} judge seeds = {st.median(stds):.3f}  "
          f"(the 'single seeded fold' variation; wrote {out}/variance_sweep.json)")
PYEOF
