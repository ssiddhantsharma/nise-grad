"""Matched-oracle-budget test: does gradient guidance beat Best-K-of-N at equal folds?

Both methods get B Boltz folds (recycling=3, num_sampling_steps=25) per run and return one
design. STE ascends the sequence by gradient; Best-K-of-N draws B sequences from STE's own
initialization distribution (0.1*normal logits, argmaxed) and keeps the best. The winners are
then scored on a HELD-OUT oracle (scripts/heldout_score.py) -- the real question is transfer,
not the Boltz number. Gradient guidance also pays a backward pass per fold, so at equal fold
COUNT it costs ~2x compute; the paper's budget is oracle QUERIES (forward evals), matched here.

Run each method in its own process (fresh JAX state, per the tracer-leak lesson):

  python matched_budget.py --method ste   --runs 3 --budget 25 --out mb.json
  python matched_budget.py --method bestn --runs 3 --budget 25 --out mb.json
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nisegrad.optimize import AA_ORDER, decode, optimize_pbind, sigmoid
from nisegrad.oracle import PbindOracle


def ste_run(oracle, feats, binder_len, budget, seed):
    logits, traj = optimize_pbind(
        oracle, feats, binder_len, steps=budget, lr=0.1,
        recycling_steps=3, straight_through=True, seed=seed)
    return decode(logits), float(traj[-1])


def bestn_run(oracle, feats, binder_len, budget, seed, key):
    """Fold+score the DISCRETE one-hot sequence, same config a single STE step folds."""
    fold_score = jax.jit(lambda seq: oracle.pbind_and_output(
        seq, feats, key, recycling_steps=3, num_sampling_steps=25)[0])
    best_seq, best_logit = None, -1e9
    for i in range(budget):
        li = 0.1 * jax.random.normal(jax.random.PRNGKey(seed * 10007 + i), (binder_len, 20))
        onehot = jax.nn.one_hot(jnp.argmax(li, -1), 20)
        logit = float(fold_score(onehot))
        if logit > best_logit:
            best_logit = logit
            best_seq = "".join(AA_ORDER[j] for j in np.asarray(onehot).argmax(-1))
    return best_seq, best_logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ste", "bestn"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=25, help="Boltz folds per run")
    ap.add_argument("--ligand", default="c1ccc(cc1)C(=O)O")  # benzoic acid
    ap.add_argument("--binder-len", type=int, default=30)
    ap.add_argument("--out", default="mb.json")
    a = ap.parse_args()

    oracle = PbindOracle(num_sampling_steps=25)   # nss>=25 for physical geometry
    feats = oracle.features_for("G" * a.binder_len, a.ligand)
    key = jax.random.PRNGKey(0)

    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    for r in range(a.runs):
        if a.method == "ste":
            seq, logit = ste_run(oracle, feats, a.binder_len, a.budget, seed=r)
        else:
            seq, logit = bestn_run(oracle, feats, a.binder_len, a.budget, seed=r, key=key)
        row = {"method": a.method, "run": r, "budget": a.budget, "ligand": a.ligand,
               "seq": seq, "boltz_logit": logit, "boltz_pbind": sigmoid(logit)}
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2))   # checkpoint after every run
        print(f"WROTE {row['method']:<6} run {r}  P(bind)={row['boltz_pbind']:.2f}  {seq}",
              flush=True)


if __name__ == "__main__":
    main()
