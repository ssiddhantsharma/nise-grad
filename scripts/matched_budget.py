"""Matched-oracle-budget sweep: does gradient guidance's edge over Best-K-of-N grow with budget?

At a budget of B Boltz folds, STE takes B gradient steps; Best-K-of-N folds B random sequences
(from STE's own init distribution) and keeps the best. Because STE's trajectory passes through
every step and Best-K-of-N's best-so-far only improves, ONE run to max(checkpoints) yields the
design at every checkpoint, so a single process per (method, seed) sweeps all budgets. Winners
are scored later on a held-out oracle (heldout_score.py); the transfer edge vs budget is the point.

One run per process (a second optimize/jit in the same process leaks a JAX tracer):

  for s in 0 1 2 3 4; do python matched_budget.py --method ste   --seed $s --out sweep.json; done
  for s in 0 1 2 3 4; do python matched_budget.py --method bestn --seed $s --out sweep.json; done
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nisegrad.optimize import AA_ORDER, decode, sigmoid
from nisegrad.oracle import PbindOracle


def ste_sweep(oracle, feats, binder_len, checkpoints, seed):
    """STE (identical to optimize_pbind) to max(checkpoints) steps; snapshot the design at each."""
    key = jax.random.PRNGKey(0)

    def loss_fn(logits):
        soft = jax.nn.softmax(logits, -1)
        hard = jax.nn.one_hot(jnp.argmax(soft, -1), 20)
        seq = soft + jax.lax.stop_gradient(hard - soft)   # forward=hard, backward=soft
        pbind, _ = oracle.pbind_and_output(seq, feats, key, recycling_steps=3)
        return -pbind, pbind

    logits = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (binder_len, 20))
    opt = optax.adam(0.1)
    state = opt.init(logits)
    step_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    cps, snaps = set(checkpoints), {}
    for i in range(1, max(checkpoints) + 1):
        (_, pbind), grad = step_fn(logits)
        updates, state = opt.update(grad, state)
        logits = optax.apply_updates(logits, updates)
        print(f"step {i:3d}  P(bind) logit {float(pbind):+.3f}", flush=True)
        if i in cps:
            snaps[i] = (decode(logits), sigmoid(float(pbind)))
    return snaps


def bestn_sweep(oracle, feats, binder_len, checkpoints, seed):
    """Fold max(checkpoints) random discrete sequences; record running-best at each checkpoint."""
    key = jax.random.PRNGKey(0)
    fold_score = jax.jit(lambda seq: oracle.pbind_and_output(
        seq, feats, key, recycling_steps=3, num_sampling_steps=25)[0])
    cps, snaps = set(checkpoints), {}
    best_seq, best_logit = None, -1e9
    for i in range(1, max(checkpoints) + 1):
        li = 0.1 * jax.random.normal(jax.random.PRNGKey(seed * 10007 + i), (binder_len, 20))
        onehot = jax.nn.one_hot(jnp.argmax(li, -1), 20)
        logit = float(fold_score(onehot))
        if logit > best_logit:
            best_logit = logit
            best_seq = "".join(AA_ORDER[j] for j in np.asarray(onehot).argmax(-1))
        if i in cps:
            snaps[i] = (best_seq, sigmoid(best_logit))
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ste", "bestn"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", default="25,100,250", help="budgets (folds) to snapshot")
    ap.add_argument("--ligand", default="c1ccc(cc1)C(=O)O")  # benzoic acid
    ap.add_argument("--binder-len", type=int, default=30)
    ap.add_argument("--out", default="sweep.json")
    a = ap.parse_args()
    checkpoints = sorted(int(x) for x in a.checkpoints.split(","))

    oracle = PbindOracle(num_sampling_steps=25)   # nss>=25 for physical geometry
    feats = oracle.features_for("G" * a.binder_len, a.ligand)
    sweep = (ste_sweep if a.method == "ste" else bestn_sweep)(
        oracle, feats, a.binder_len, checkpoints, a.seed)

    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    for budget, (seq, pbind) in sweep.items():
        rows.append({"method": a.method, "seed": a.seed, "budget": budget,
                     "ligand": a.ligand, "seq": seq, "boltz_pbind": pbind})
        print(f"WROTE {a.method:<6} seed {a.seed} budget {budget:3d}  P(bind)={pbind:.2f}  {seq}",
              flush=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
