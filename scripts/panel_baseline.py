"""Matched-budget gradient-free baseline over the ligand panel (reviewer ask: does the gradient
beat gradient-free search at equal fold budget?). For each ligand in ligand_panel.json: real /
scramble / random anchors + Best-K-of-N (bestn) and O3 latent-BO (o3) designs at fold budget 25,
scored on held-out Protenix. Mirrors panel_run.py (the STE arm); compare bestn / o3 vs STE vs the
real anchor per ligand. Env: CUDA_VISIBLE_DEVICES, PROTENIX_DIR, OUT.
"""

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
DATA = REPO / "scripts/data"
OUT = Path(os.environ.get("OUT", "/tmp/panel_baseline")); OUT.mkdir(parents=True, exist_ok=True)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
AA = "ARNDCQEGHILKMFPSTWYV"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:20]


def protenix(J):
    name = J.stem
    subprocess.run([PY, str(REPO / "scripts/protenix_score.py"), "build", "--in", str(J),
                    "--input-json", str(OUT / f"{name}_in.json")], check=True)
    subprocess.run(["bash", "-c",
                    (f'cd "{os.environ["PROTENIX_DIR"]}" && CUDA_VISIBLE_DEVICES={GPU} .venv/bin/protenix '
                     f'pred -i {OUT}/{name}_in.json -o {OUT}/{name}_out -s 101 -n protenix-v2 '
                     f'--use_msa false --use_default_params true')], check=True)
    subprocess.run([PY, str(REPO / "scripts/protenix_score.py"), "parse", "--in", str(J),
                    "--outdir", str(OUT / f"{name}_out"), "--seed", "101"], check=True)


def anchor_rows(seq, smi, name):
    random.seed(0)
    scr = list(seq); random.shuffle(scr)
    rnd = "".join(random.choice(AA) for _ in seq)
    return [{"method": "anchor_real", "seed": 0, "budget": 0, "ligand": smi, "seq": seq, "note": name},
            {"method": "anchor_scramble", "seed": 0, "budget": 0, "ligand": smi, "seq": "".join(scr)},
            {"method": "anchor_random", "seed": 0, "budget": 0, "ligand": smi, "seq": rnd}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lig-start", type=int, default=0)
    ap.add_argument("--lig-end", type=int, default=999)
    ap.add_argument("--bestn-seeds", type=int, default=8)
    ap.add_argument("--o3-seeds", type=int, default=5)
    a = ap.parse_args()

    panel = json.loads((DATA / "ligand_panel.json").read_text())[a.lig_start:a.lig_end]
    for p in panel:
        J = OUT / f"lig_{slug(p['ligand_name'])}_baseline.json"
        J.write_text(json.dumps(anchor_rows(p["anchor_seq"], p["ligand"], p["anchor_name"]), indent=2))
        L = len(p["anchor_seq"])
        for method, n in [("bestn", a.bestn_seeds), ("o3", a.o3_seeds)]:
            for s in range(n):
                subprocess.run([PY, str(REPO / "scripts/matched_budget.py"), "--method", method,
                                "--seed", str(s), "--checkpoints", "25", "--binder-len", str(L),
                                "--ligand", p["ligand"], "--out", str(J)],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU}, check=True)
        protenix(J)
        print(f"LIGAND_DONE {p['ligand_name']}", flush=True)
    print("BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
