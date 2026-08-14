"""Matched-oracle-budget sweep: does gradient guidance's edge over Best-K-of-N grow with budget?

At a budget of B Boltz folds, STE takes B gradient steps; Best-K-of-N folds B random sequences
(from STE's own init distribution) and keeps the best. Because STE's trajectory passes through
every step and Best-K-of-N's best-so-far only improves, ONE run to max(checkpoints) yields the
design at every checkpoint, so a single process per (method, seed) sweeps all budgets. Winners
are scored later on a held-out oracle (heldout_score.py); the transfer edge vs budget is the point.

Methods: ste (gradient), bestn (Best-K-of-N sampling), o3 (latent Bayesian optimization, an
adapted Kalisz et al. 2026 baseline). One run per process (a second optimize/jit in the same
process leaks a JAX tracer):

  for s in 0 1 2 3 4; do python matched_budget.py --method ste   --seed $s --out sweep.json; done
  for s in 0 1 2 3 4; do python matched_budget.py --method bestn --seed $s --out sweep.json; done
  for s in 0 1 2 3 4; do python matched_budget.py --method o3    --seed $s --out sweep.json; done
"""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nisegrad.optimize import AA_ORDER, decode, sigmoid
from nisegrad.oracle import PbindOracle


def load_ligandmpnn(ckpt, ref_dir):
    """Load a LigandMPNN checkpoint into jligandmpnn (JAX) via the reference torch module."""
    import sys

    import torch
    sys.path.insert(0, ref_dir)
    import ligmpnn_model as ref
    from jligandmpnn.model import LigandMPNN
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=ck["num_edges"],
                        atom_context_num=ck["atom_context_num"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return LigandMPNN.from_torch(m)


def ste_sweep(oracle, feats, binder_len, checkpoints, seed, reg=None, mpnn_weight=0.0):
    """STE (as in optimize_pbind) to max(checkpoints) steps; snapshot the design at each. reg is
    an optional ligand-aware LigandMPNN regularizer (NLL of the soft sequence given the live
    predicted structure), added at mpnn_weight to counter the affinity head's reward-hacking."""
    key = jax.random.PRNGKey(0)

    def loss_fn(logits):
        soft = jax.nn.softmax(logits, -1)
        hard = jax.nn.one_hot(jnp.argmax(soft, -1), 20)
        seq = soft + jax.lax.stop_gradient(hard - soft)   # forward=hard, backward=soft
        pbind, output = oracle.pbind_and_output(seq, feats, key, recycling_steps=3)
        loss = -pbind
        if reg is not None:
            loss = loss + mpnn_weight * reg(seq, output, key)[0]
        return loss, pbind

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


def o3_sweep(oracle, feats, binder_len, checkpoints, seed, n_seed=8, n_dims=6, pool=512):
    """O3-style latent Bayesian optimization (Kalisz et al. 2026), adapted to sequence design.

    O3 builds a low-dimensional subspace of a generative model's output space from a few top
    seeds and runs an off-the-shelf optimizer over it. Kalisz optimize structure-generation
    latents for conformation recovery; for de-novo design the natural output space is the binder
    logit simplex, so we build the subspace by PCA of n_seed random seed-logit vectors and run a
    Gaussian-process / expected-improvement loop over it. Gradient-free, same fold budget as STE
    and Best-K-of-N (forward queries). Records running-best at each checkpoint."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    key = jax.random.PRNGKey(0)
    fold_score = jax.jit(lambda oh: oracle.pbind_and_output(
        oh, feats, key, recycling_steps=3, num_sampling_steps=25)[0])
    rng = np.random.default_rng(seed)
    budget, cps, snaps = max(checkpoints), set(checkpoints), {}
    best_p, best_s = -1e9, None

    def fold_logits(flat):
        oh = jax.nn.one_hot(jnp.argmax(jnp.asarray(flat).reshape(binder_len, 20), -1), 20)
        return float(fold_score(oh)), "".join(AA_ORDER[j] for j in np.asarray(oh).argmax(-1))

    def observe(p, s, n):
        nonlocal best_p, best_s
        if p > best_p:
            best_p, best_s = p, s
        if n in cps:
            snaps[n] = (best_s, sigmoid(best_p))

    X = np.stack([0.1 * rng.standard_normal(binder_len * 20) for _ in range(n_seed)])
    y, n = [], 0
    for flat in X:
        p, s = fold_logits(flat)
        y.append(p)
        n += 1
        observe(p, s, n)
    mean = X.mean(0)
    basis = np.linalg.svd(X - mean, full_matrices=False)[2][:n_dims]   # [n_dims, D]
    coords = (X - mean) @ basis.T
    lo, hi = coords.min(0) - 2, coords.max(0) + 2
    while n < budget:
        gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0), normalize_y=True, alpha=1e-4)
        gp.fit(coords, np.array(y))
        cand = rng.uniform(lo, hi, size=(pool, n_dims))
        mu, sd = gp.predict(cand, return_std=True)
        z = (mu - max(y)) / np.maximum(sd, 1e-9)
        ei = (mu - max(y)) * norm.cdf(z) + sd * norm.pdf(z)
        c = cand[int(np.argmax(ei))]
        p, s = fold_logits(mean + c @ basis)
        coords, y, n = np.vstack([coords, c]), y + [p], n + 1
        observe(p, s, n)
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ste", "bestn", "o3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", default="25,100,250", help="budgets (folds) to snapshot")
    ap.add_argument("--ligand", default="c1ccc(cc1)C(=O)O")  # benzoic acid
    ap.add_argument("--binder-len", type=int, default=30)
    ap.add_argument("--mpnn-weight", type=float, default=0.0,
                    help="ste only: weight of the LigandMPNN prior (env LIGANDMPNN_CKPT, "
                         "LIGMPNN_MODEL_DIR); 0 = plain STE")
    ap.add_argument("--out", default="sweep.json")
    a = ap.parse_args()
    checkpoints = sorted(int(x) for x in a.checkpoints.split(","))

    oracle = PbindOracle(num_sampling_steps=25)   # nss>=25 for physical geometry
    feats = oracle.features_for("G" * a.binder_len, a.ligand)

    if a.method == "ste":
        reg = None
        if a.mpnn_weight > 0:
            from nisegrad.boltz_ligand import build_boltz_regularizer
            model = load_ligandmpnn(os.environ["LIGANDMPNN_CKPT"],
                                    os.environ["LIGMPNN_MODEL_DIR"])
            reg = build_boltz_regularizer(model, feats, frozen_output=None)
        sweep = ste_sweep(oracle, feats, a.binder_len, checkpoints, a.seed,
                          reg=reg, mpnn_weight=a.mpnn_weight)
    else:
        sweep = {"bestn": bestn_sweep, "o3": o3_sweep}[a.method](
            oracle, feats, a.binder_len, checkpoints, a.seed)

    label = a.method if a.mpnn_weight == 0 else f"ste_w{a.mpnn_weight:g}"
    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    for budget, (seq, pbind) in sweep.items():
        rows.append({"method": label, "seed": a.seed, "budget": budget,
                     "ligand": a.ligand, "seq": seq, "boltz_pbind": pbind})
        print(f"WROTE {label:<8} seed {a.seed} budget {budget:3d}  P(bind)={pbind:.2f}  {seq}",
              flush=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
