#!/usr/bin/env bash
# Run 2: RFd3 de-novo backbone arm for apixaban. RFd3 backbones (bury APX) -> LigandMPNN sequences ->
# held-out Protenix interface pTM (nise-grad scorer, seed 101, single-seq), directly comparable to the
# plateau (de-novo STE ~0.45) and rescue (real backbone ~0.83). All on GPU 2.
set -uo pipefail
# Portable paths. REPO is this checkout; the RFd3 backbone pipeline and Protenix judge are external
# installs the user points at (see README "Reproducibility tiers"): CHEM_FAP_DIR and PROTENIX_DIR.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHEM_FAP_DIR="${CHEM_FAP_DIR:?set CHEM_FAP_DIR to your RFd3 / chemistry_fap pipeline checkout}"
SMI="COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N"
cd "$CHEM_FAP_DIR"
export DYE=apix DYE_SMILES="$SMI" LIG_RESNAME=APX RUN_NAME=apix_rfd3 BINDER_LEN=117 \
       N_DESIGNS=24 N_SEQS=4 ALLOWED_GPUS=2 CUDA_VISIBLE_DEVICES=2 HBOND_LIGAND_ATOM= FOLD_CLOSED=0
source paths.sh

echo "[1/4] rfd3 backbones ($N_DESIGNS)"; bash scripts/rfd3.sh || { echo "RFD3 FAILED"; exit 1; }
echo "[2/4] ligandmpnn sequences ($N_SEQS/backbone)"; bash scripts/ligandmpnn.sh || { echo "MPNN FAILED"; exit 1; }

RUN="$CHEM_FAP_DIR/runs/apix_rfd3"
J="$REPO/scripts/data/rfd3_arm_designs.json"
echo "[3/4] build designs.json"
python3 - "$RUN/ligandmpnn/seqs" "$SMI" "$J" <<'PY'
import sys, json, glob, os
seqdir, smi, out = sys.argv[1:4]
rows = []
for fa in sorted(glob.glob(os.path.join(seqdir, "*.fa"))):
    lines = [l.strip() for l in open(fa) if l.strip()]
    bb = os.path.basename(fa)[:-3]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            hdr, seq = lines[i], (lines[i + 1] if i + 1 < len(lines) else "")
            if "id=" in hdr and seq and not seq.startswith(">"):
                sid = hdr.split("id=")[1].split(",")[0].strip()
                rows.append({"method": "rfd3_mpnn", "seed": int(sid), "budget": 0,
                             "ligand": smi, "seq": seq, "note": bb})
            i += 2
        else:
            i += 1
json.dump(rows, open(out, "w"), indent=2)
print("wrote", len(rows), "designs ->", out)
PY

echo "[4/4] protenix score (held-out judge, nise-grad)"
cd "$REPO"
export PROTENIX_DIR="${PROTENIX_DIR:?set PROTENIX_DIR to your Protenix checkout}"
OUT="$REPO/scripts/data/rfd3_arm"; mkdir -p "$OUT"
.venv/bin/python scripts/protenix_score.py build --in "$J" --input-json "$OUT/rfd3_arm_in.json"
( cd "$PROTENIX_DIR" && CUDA_VISIBLE_DEVICES=2 .venv/bin/protenix pred -i "$OUT/rfd3_arm_in.json" \
    -o "$OUT/rfd3_arm_out" -s 101 -n protenix-v2 --use_msa false --use_default_params true )
.venv/bin/python scripts/protenix_score.py parse --in "$J" --outdir "$OUT/rfd3_arm_out" --seed 101
echo "RFD3_ARM_DONE  designs=$J  scored=$OUT"
