"""Held-out oracle: score matched_budget designs on jopendde ipTM (independent of Boltz).

The transfer verdict. Reads matched_budget's JSON, folds each design's (sequence, ligand) with
jopendde, prints ipTM per design and the per-method mean, and writes the ipTM back into the JSON.
A real, experimentally-validated binder scores ~0.98 on this oracle (see the independence gate),
so it discriminates; the question is whether STE designs beat Best-K-of-N designs on an oracle
neither method optimized. Protenix-v2 is the confirmatory second held-out oracle (separate CLI).
"""

import argparse
import json
from pathlib import Path

import jax
from jopendde.inference import Predictor, predict, summarize


def iptm(p, key, seq, lig):
    spec = [{"name": "x", "modelSeeds": [0], "sequences": [
        {"proteinChain": {"sequence": seq, "count": 1}},
        {"ligand": {"ligand": lig, "count": 1}}]}]
    inp = p.featurize(spec)
    pred = predict(p.model, inp.feat, key, n_cycle=10, n_sample=1, n_step=200)
    return float(summarize(inp.feat, pred, p.summary_params, n_recycle=10)[0].get("iptm", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="mb.json")
    a = ap.parse_args()
    rows = json.loads(Path(a.inp).read_text())
    p = Predictor.from_checkpoint()
    key = jax.random.key(0)

    by = {}
    for row in rows:
        ip = iptm(p, key, row["seq"], row["ligand"])
        row["jopendde_iptm"] = ip
        by.setdefault(row["method"], []).append(ip)
        Path(a.inp).write_text(json.dumps(rows, indent=2))
        print(f"HELDOUT {row['method']:<6} run {row['run']}  boltz {row['boltz_pbind']:.2f}  "
              f"jopendde_iptm {ip:.2f}  {row['seq']}", flush=True)
    print("---")
    for m, xs in by.items():
        print(f"MEAN {m:<6} jopendde_iptm {sum(xs) / len(xs):.2f} (n={len(xs)})", flush=True)


if __name__ == "__main__":
    main()
