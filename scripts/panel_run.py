"""Ligand-panel runner. Two modes:
  kdval    fold the real de-novo binders (kdval.json) on Protenix -> score vs measured pKd (B).
  ligands  per ligand in ligand_panel.json: real anchor + STE de-novo designs -> Protenix (A).
Reuses matched_budget.py (STE, one subprocess/seed for the tracer bug) + protenix_score.py.
Env: CUDA_VISIBLE_DEVICES, PROTENIX_DIR, OUT.
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
OUT = Path(os.environ.get("OUT", "/tmp/panel_run")); OUT.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("mode", choices=["kdval", "ligands"])
    ap.add_argument("--lig-start", type=int, default=0)
    ap.add_argument("--lig-end", type=int, default=999)
    ap.add_argument("--nseeds", type=int, default=10)
    a = ap.parse_args()

    if a.mode == "kdval":
        rows = json.loads((DATA / "kdval.json").read_text())
        designs = [{"method": "real_binder", "seed": i, "budget": 0, "ligand": r["ligand"],
                    "seq": r["seq"], "pKd": r["pKd"], "note": r["name"], "ligand_name": r["ligand_name"]}
                   for i, r in enumerate(rows)]
        J = OUT / "kdval.json"
        J.write_text(json.dumps(designs, indent=2))
        protenix(J)
        print("KDVAL_DONE", flush=True)
        return

    panel = json.loads((DATA / "ligand_panel.json").read_text())[a.lig_start:a.lig_end]
    for p in panel:
        J = OUT / f"lig_{slug(p['ligand_name'])}.json"
        J.write_text(json.dumps(anchor_rows(p["anchor_seq"], p["ligand"], p["anchor_name"]), indent=2))
        for s in range(a.nseeds):
            subprocess.run([PY, str(REPO / "scripts/matched_budget.py"), "--method", "ste", "--seed", str(s),
                            "--checkpoints", "25", "--binder-len", str(len(p["anchor_seq"])),
                            "--ligand", p["ligand"], "--out", str(J)],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU}, check=True)
        protenix(J)
        print(f"LIGAND_DONE {p['ligand_name']}", flush=True)
    print("LIGANDS_DONE", flush=True)


if __name__ == "__main__":
    main()
