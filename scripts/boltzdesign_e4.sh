#!/usr/bin/env bash
# E4 — BoltzDesign1 head-to-head on the small-molecule panel.
#
# Runs BoltzDesign1 (native design, --run_ligandmpnn False) per ligand, extracts the
# binder sequence from the design .cif, and scores it on the SAME held-out Protenix-v2
# judge as the gradient / RFd3 arms. Reports per-ligand best composition-passing
# interface pTM vs gradient (0.44/0.66 mean/best), RFd3 (0.79/0.89), real binder (0.93).
#
# BoltzDesign inverts the SAME Boltz model the gradient uses, but targets the distogram
# (structure) rather than the affinity head -- so this isolates the paper's thesis
# (structure inversion vs affinity inversion). Native design is used (not LigandMPNN):
# BoltzDesign emits backbone AND sequence, and scoring every design on OUR held-out judge
# is the fair comparison, bypassing BoltzDesign's internal iPTM filter.
#
# SETUP (validated per-step on apixaban 2026-09-24, bizon A6000; apixaban 1-sample -> 0.52):
# BoltzDesign1 needs, beyond its repo checkout ($BD_DIR):
#   - env with torch built for the driver's CUDA (torch==2.6.0+cu124 for CUDA 12.9; the
#     default install pulls a too-new cu130 build) ....... upstream issue yehlincho/BoltzDesign1#38
#   - IPython in the env (upstream setup.sh installs ipykernel; a bare env misses it)
#   - ~/.boltz/boltz1_conf.ckpt + ccd.pkl as REAL files (not dangling symlinks)
#   - $BD_DIR/env/bin on PATH (boltzdesign shells out to the `boltz` CLI)
#   - for --run_ligandmpnn True only: seed {work_dir}/LigandMPNN (config + model_params),
#     or use --work_dir = repo root ............ fixed upstream in yehlincho/BoltzDesign1#37
#
# Per-design cost ~25 min (three ~75-epoch stages). 14 x 8 ~= 46h on one A6000 -> for the
# full 40-budget RFd3-matched run, offload to an H100.
#
#   BD_DIR=~/siddhant/tools/BoltzDesign1 PROTENIX_DIR=~/Protenix CUDA_VISIBLE_DEVICES=0 \
#     N=8 LIGS="apixaban cortisol DFHBI camptothecin amantadine" bash scripts/boltzdesign_e4.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BD_DIR="${BD_DIR:?set BD_DIR to the BoltzDesign1 checkout}"
PROTENIX_DIR="${PROTENIX_DIR:?set PROTENIX_DIR to the Protenix judge checkout}"
NG_PY="$REPO/.venv/bin/python"          # nise-grad venv: gemmi + protenix_score.py
BD_PY="$BD_DIR/env/bin/python"
PS="$REPO/scripts/protenix_score.py"
PANEL="$REPO/scripts/data/ligand_panel.json"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
N="${N:-8}"
SEED="${SEED:-101}"
OUT="${OUT:-/tmp/e4_boltzdesign}"; mkdir -p "$OUT"

mapfile -t ROWS < <("$NG_PY" - "$PANEL" "${LIGS:-}" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
want = set(sys.argv[2].split()) if len(sys.argv) > 2 and sys.argv[2] else None
wl = {w.lower() for w in want} if want else None
for e in d:
    if e.get("source") == "Cell2026_Rhobin":
        continue
    slug = e["ligand_name"].lower().replace(" ", "").replace(",", "").replace("'", "")
    if wl and e["ligand_name"] not in want and slug not in wl:
        continue
    print(f'{slug}\t{len(e["anchor_seq"])}\t{e["ligand"]}')
PYEOF
)

DESIGNS="$OUT/e4_designs.json"; echo "[]" > "$DESIGNS"

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r slug L SM <<< "$row"
  echo "== BoltzDesign $slug (len $L, $N samples) on GPU $GPU =="
  ( cd "$BD_DIR" && PATH="$BD_DIR/env/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPU" "$BD_PY" boltzdesign.py \
      --target_name "$slug" --target_type small_molecule --input_type custom \
      --custom_target_input "$SM" --custom_target_ids B --binder_id A \
      --length_min "$L" --length_max "$L" --design_samples "$N" \
      --use_msa False --run_boltz_design True --run_ligandmpnn False \
      --run_alphafold False --run_rosetta False \
      --gpu_id "$GPU" --work_dir "$OUT/$slug" --suffix e4 ) || { echo "  $slug FAILED (surfacing, not skipping)"; continue; }

  # extract the binder (chain A) sequence from every design .cif -> append design rows
  "$NG_PY" - "$OUT/$slug" "$slug" "$SM" "$DESIGNS" <<'PYEOF'
import glob, json, sys, gemmi
from collections import Counter
out_slug, slug, smiles, designs_path = sys.argv[1:5]
cifs = [c for c in glob.glob(f"{out_slug}/outputs/**/results_final/**/*_model_0.cif", recursive=True)
        if "_apo" not in c]
rows = json.load(open(designs_path))
for cif in sorted(cifs):
    s = gemmi.read_structure(cif)
    seq = ""
    for ch in s[0]:
        q = ch.get_polymer().make_one_letter_sequence()
        if q and len(q) > 10:
            seq = q; break
    if not seq:
        continue
    maxaa = Counter(seq).most_common(1)[0][1] / len(seq)
    rows.append({"method": "boltzdesign", "seed": 0, "budget": 0, "ligand": smiles,
                 "seq": seq, "maxaa": round(maxaa, 3), "note": slug, "ligand_name": slug})
json.dump(rows, open(designs_path, "w"), indent=1)
print(f"  {slug}: extracted {len(cifs)} design sequence(s)")
PYEOF
done

echo "== scoring all BoltzDesign designs on the held-out Protenix judge (seed $SEED) =="
"$NG_PY" "$PS" build --in "$DESIGNS" --input-json "$OUT/e4_in.json"
( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/protenix pred \
    -i "$OUT/e4_in.json" -o "$OUT/e4_out" -s "$SEED" -n protenix-v2 \
    --use_msa false --use_default_params true > "$OUT/e4_pred.log" 2>&1 )
"$NG_PY" "$PS" parse --in "$DESIGNS" --outdir "$OUT/e4_out" --seed "$SEED"

# aggregate: per-ligand best composition-passing interface pTM (maxAA < 0.35), vs the arms
"$NG_PY" - "$DESIGNS" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
by = {}
for r in d:
    v = r.get("protenix_iptm")
    if v is None:
        continue
    by.setdefault(r["ligand_name"], []).append((v, r.get("maxaa", 1.0)))
print(f"\n{'ligand':<16}{'best_pass':>10}{'n':>4}{'polyX_disq':>12}")
bests = []
for lig, vs in sorted(by.items()):
    passing = [v for v, m in vs if m < 0.35]
    disq = sum(1 for _, m in vs if m >= 0.35)
    b = max(passing) if passing else 0.0
    bests.append(b)
    print(f"{lig:<16}{b:>10.3f}{len(vs):>4}{disq:>12}")
if bests:
    print(f"\nBoltzDesign panel mean best-pass = {sum(bests)/len(bests):.3f}")
    print("compare: gradient 0.44/0.66 | RFd3 0.79/0.89 | real 0.93")
PYEOF
echo "ALLDONE  designs+scores in $DESIGNS"
