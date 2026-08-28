"""Affinity-predictor leaderboard on de-novo protein->small-molecule binders (experiment C).

Do dedicated affinity heads (Boltz-2, Nesso-1) beat trivial baselines (MW, cLogP) at ranking
measured KD on de-novo designed binders? Mirrors the OpenBind EV-A71 leaderboard on a de-novo
distribution. 37 rows (kdval.json): protein sequence + ligand SMILES + measured pKd.

Modes:
  boltz   real boltz CLI, affinity prediction, single-sequence (no MSA), recycling 3
  nesso   recursionpharma/nesso, seq+SMILES YAML (ESM, no MSA)
  score   Spearman rho of every method + MW/cLogP baselines vs measured pKd

Env: CUDA_VISIBLE_DEVICES, OUT (persistent working dir), BOLTZ (boltz exe), NESSO (nesso exe).
Both models emit affinity_pred_value; OpenBind pK conversion: pK = 6 - affinity_pred_value.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "scripts/data"
OUT = Path(os.environ.get("OUT", "/tmp/affinity_leaderboard")); OUT.mkdir(parents=True, exist_ok=True)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
BOLTZ = os.environ.get("BOLTZ", str(REPO / ".venv/bin/boltz"))
NESSO = os.environ.get("NESSO", "nesso")


def rows():
    return json.loads((DATA / os.environ.get("INPUT", "kdval.json")).read_text())


def yaml_for(seq, smiles, msa_empty):
    msa = "      msa: empty\n" if msa_empty else ""
    return (f"sequences:\n  - protein:\n      id: A\n      sequence: {seq}\n{msa}"
            f"  - ligand:\n      id: B\n      smiles: '{smiles}'\n"
            f"properties:\n  - affinity:\n      binder: B\n")


def write_yamls(subdir, msa_empty, limit):
    d = OUT / subdir; d.mkdir(exist_ok=True)
    for i, r in enumerate(rows()[:limit]):
        (d / f"c{i:02d}.yaml").write_text(yaml_for(r["seq"], r["ligand"], msa_empty))
    return d


def run_boltz(limit):
    ydir = write_yamls("boltz_yaml", True, limit)
    subprocess.run([BOLTZ, "predict", str(ydir), "--out_dir", str(OUT / "boltz_out"),
                    "--recycling_steps", "3", "--diffusion_samples", "1", "--override",
                    "--no_kernels", "--devices", "1"],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU}, check=True)
    out = []
    pred = OUT / "boltz_out" / "boltz_results_boltz_yaml" / "predictions"
    for i, r in enumerate(rows()[:limit]):
        aff = pred / f"c{i:02d}" / f"affinity_c{i:02d}.json"
        conf = pred / f"c{i:02d}" / f"confidence_c{i:02d}_model_0.json"
        if not aff.exists():
            print(f"MISSING boltz c{i:02d} ({r['ligand_name']})", flush=True); continue
        a = json.loads(aff.read_text())
        c = json.loads(conf.read_text()) if conf.exists() else {}
        v = a.get("affinity_pred_value")
        out.append({"name": r["name"], "ligand_name": r["ligand_name"], "pKd": r["pKd"],
                    "boltz_pK": 6 - v if v is not None else None,
                    "boltz_pbind": a.get("affinity_probability_binary"), "boltz_iptm": c.get("iptm")})
    (OUT / "boltz_scored.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE boltz_scored.json  n={len(out)}", flush=True)


def run_nesso(limit):
    ydir = write_yamls("nesso_yaml", False, limit)
    subprocess.run([NESSO, "predict", str(ydir), "--out_dir", str(OUT / "nesso_out"),
                    "--override", "--devices", "1"],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU}, check=True)
    out = []
    pred = OUT / "nesso_out" / "predictions"
    for i, r in enumerate(rows()[:limit]):
        aff = pred / f"c{i:02d}" / "affinity.json"
        if not aff.exists():
            print(f"MISSING nesso c{i:02d} ({r['ligand_name']})", flush=True); continue
        a = json.loads(aff.read_text())
        v = a.get("affinity_pred_value")
        out.append({"name": r["name"], "ligand_name": r["ligand_name"], "pKd": r["pKd"],
                    "nesso_pK": 6 - v if v is not None else None,
                    "nesso_pbind": a.get("affinity_probability_binary")})
    (OUT / "nesso_scored.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE nesso_scored.json  n={len(out)}", flush=True)


def _rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for i, k in enumerate(o):
        r[k] = i
    return r


def run_protenix(limit):
    """Protenix-v2 on all rows, same params as the B run (single-seq, seed 101). Needs PROTENIX_DIR."""
    pdir = os.environ["PROTENIX_DIR"]
    J = OUT / "protenix_scored.json"; J.write_text(json.dumps(rows()[:limit], indent=2))
    sc = str(REPO / "scripts/protenix_score.py")
    subprocess.run([REPO / ".venv/bin/python", sc, "build", "--in", str(J),
                    "--input-json", str(OUT / "px_in.json")], check=True)
    subprocess.run(["bash", "-c",
                    f'cd "{pdir}" && CUDA_VISIBLE_DEVICES={GPU} .venv/bin/protenix pred '
                    f'-i {OUT}/px_in.json -o {OUT}/px_out -s 101 -n protenix-v2 '
                    f'--use_msa false --use_default_params true'], check=True)
    subprocess.run([REPO / ".venv/bin/python", sc, "parse", "--in", str(J),
                    "--outdir", str(OUT / "px_out"), "--seed", "101"], check=True)
    print(f"WROTE protenix_scored.json  n={len(json.loads(J.read_text()))}", flush=True)


def spearman(xs, ys):
    rx, ry = _rank(xs), _rank(ys); n = len(xs)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy)


def rmse(xs, ys):
    return (sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5


def bootstrap_ci(xs, ys, stat, n_boot=2000, seed=0):
    import random
    rng = random.Random(seed); n = len(xs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(stat([xs[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def score():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    def key(r):
        return (r["name"], r["ligand_name"])        # name alone collides (PARP: 1 scaffold, 4 ligands)
    merged = {key(r): dict(r) for r in rows()}
    assert len(merged) == len(rows()), f"key collision: {len(merged)} unique vs {len(rows())} rows"
    for f in ("boltz_scored.json", "nesso_scored.json", "protenix_scored.json"):
        p = OUT / f
        if p.exists():
            for r in json.loads(p.read_text()):
                merged[key(r)].update(r)
    for r in merged.values():
        m = Chem.MolFromSmiles(r["ligand"])
        r["mw"] = Descriptors.MolWt(m) if m else None
        r["clogp"] = Crippen.MolLogP(m) if m else None
    d = list(merged.values())
    print(f"{'method':22s}{'rho [95% CI]':>22s}{'pearson':>9s}{'RMSE':>7s}{'n':>5s}")
    for key in ("boltz_pK", "boltz_pbind", "boltz_iptm", "nesso_pK", "nesso_pbind", "clogp", "mw"):
        vals = [(r[key], r["pKd"]) for r in d if r.get(key) is not None]
        if len(vals) <= 2:
            continue
        xs, ys = map(list, zip(*vals))
        rho = spearman(xs, ys); lo, hi = bootstrap_ci(xs, ys, spearman)
        # RMSE only meaningful for calibrated pK predictions
        rm = f"{rmse(xs, ys):.2f}" if key.endswith("_pK") else "-"
        print(f"{key:22s}{f'{rho:+.2f} [{lo:+.2f},{hi:+.2f}]':>22s}{pearson(xs, ys):>+9.2f}{rm:>7s}{len(vals):>5d}")
    (OUT / "leaderboard_scored.json").write_text(json.dumps(d, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["boltz", "nesso", "protenix", "score"])
    ap.add_argument("--limit", type=int, default=37)
    a = ap.parse_args()
    {"boltz": lambda: run_boltz(a.limit), "nesso": lambda: run_nesso(a.limit),
     "protenix": lambda: run_protenix(a.limit), "score": score}[a.mode]()


if __name__ == "__main__":
    main()
