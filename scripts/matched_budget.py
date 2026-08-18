"""Matched-oracle-budget sweep: does gradient guidance's edge over Best-K-of-N grow with budget?

At a budget of B Boltz folds, STE takes B gradient steps; Best-K-of-N folds B random sequences
(from STE's own init distribution) and keeps the best. Because STE's trajectory passes through
every step and Best-K-of-N's best-so-far only improves, ONE run to max(checkpoints) yields the
design at every checkpoint, so a single process per (method, seed) sweeps all budgets. Winners
are scored later on a held-out oracle (protenix_score.py); the transfer edge vs budget is the point.

Methods: ste (gradient), bestn (Best-K-of-N sampling), o3 (latent Bayesian optimization, an
adapted Kalisz et al. 2026 baseline). One run per process (a second optimize/jit in the same
process leaks a JAX tracer):

  for s in 0 1 2 3 4; do python matched_budget.py --method ste   --seed $s --out sweep.json; done
  for s in 0 1 2 3 4; do python matched_budget.py --method bestn --seed $s --out sweep.json; done
  for s in 0 1 2 3 4; do python matched_budget.py --method o3    --seed $s --out sweep.json; done
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mosaic.losses.structure_prediction import BinderTargetContact

from nisegrad.optimize import AA_ORDER, decode, foldability, interface_pae, ptm_energy, sigmoid
from nisegrad.oracle import PbindOracle


def ste_sweep(oracle, feats, binder_len, checkpoints, seed, confidence_weight=0.0, init_seq=None,
              contact_weight=0.0, ptm_energy_obj=False, pocket_scaffold=False):
    """STE (as in optimize_pbind) to max(checkpoints) steps; snapshot the design at each.
    confidence_weight>0 adds the binder-ligand interface PAE (a second Boltz head), so the
    design must form a confident predicted interface, not just fool the affinity head.
    contact_weight>0 adds mosaic's BinderTargetContact on the Boltz distogram (binder residue ->
    ligand atom, <8A), which explicitly forces a real interface rather than a gamed affinity head.
    ptm_energy_obj replaces the affinity head with pTMEnergy (BindEnergyCraft, Eq 8): a dense,
    LogSumExp interface energy over the pAE logits that resists the reward-hacking of the max-based
    affinity head. init_seq (a real scaffold sequence) biases the starting logits so STE refines a
    real fold rather than hallucinating from noise; per-seed noise still varies the trajectory."""
    key = jax.random.PRNGKey(0)
    contact = BinderTargetContact(contact_distance=8.0) if contact_weight else None
    # pocket-then-scaffold (L-Caliby): concentrate contact on the top-12 pocket residues, fold
    # the rest. Two objectives, split by role, instead of one pressure spread over all residues.
    pocket = BinderTargetContact(contact_distance=8.0, paratope_size=12) if pocket_scaffold else None

    def loss_fn(logits):
        soft = jax.nn.softmax(logits, -1)
        hard = jax.nn.one_hot(jnp.argmax(soft, -1), 20)
        seq = soft + jax.lax.stop_gradient(hard - soft)   # forward=hard, backward=soft
        pbind, output = oracle.pbind_and_output(seq, feats, key, recycling_steps=3)
        loss = ptm_energy(output) if ptm_energy_obj else -pbind
        if confidence_weight:
            loss = loss + confidence_weight * interface_pae(output)
        if contact_weight:
            loss = loss + contact_weight * contact(seq, output, key)[0]
        if pocket_scaffold:
            loss = loss + 0.1 * pocket(seq, output, key)[0] + 0.1 * foldability(output)
        return loss, pbind

    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (binder_len, 20))
    if init_seq is not None:
        assert len(init_seq) == binder_len, f"init_seq len {len(init_seq)} != binder_len {binder_len}"
        idx = jnp.asarray([AA_ORDER.index(c) for c in init_seq])
        # bias 2.0: argmax starts at the scaffold, but 25 Adam steps can still flip key positions
        logits = 2.0 * jax.nn.one_hot(idx, 20) + noise
    else:
        logits = noise
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
    ap.add_argument("--confidence-weight", type=float, default=0.0,
                    help="ste only: weight of the interface-PAE confidence term (0 = plain STE)")
    ap.add_argument("--init-seq", default=None,
                    help="ste only: real scaffold sequence to initialize from (else random)")
    ap.add_argument("--contact-weight", type=float, default=0.0,
                    help="ste only: weight of mosaic BinderTargetContact (distogram, 0 = off)")
    ap.add_argument("--ptm-energy", action="store_true",
                    help="ste only: optimize pTMEnergy (BindEnergyCraft) instead of the affinity head")
    ap.add_argument("--pocket-scaffold", action="store_true",
                    help="ste only: L-Caliby pocket-then-scaffold (pocket contact + scaffold pLDDT)")
    ap.add_argument("--out", default="sweep.json")
    a = ap.parse_args()
    checkpoints = sorted(int(x) for x in a.checkpoints.split(","))

    oracle = PbindOracle(num_sampling_steps=25)   # nss>=25 for physical geometry
    feats = oracle.features_for("G" * a.binder_len, a.ligand)
    if a.method == "ste":
        sweep = ste_sweep(oracle, feats, a.binder_len, checkpoints, a.seed,
                          confidence_weight=a.confidence_weight, init_seq=a.init_seq,
                          contact_weight=a.contact_weight, ptm_energy_obj=a.ptm_energy,
                          pocket_scaffold=a.pocket_scaffold)
    else:
        sweep = {"bestn": bestn_sweep, "o3": o3_sweep}[a.method](
            oracle, feats, a.binder_len, checkpoints, a.seed)

    label = a.method
    if a.pocket_scaffold:
        label = "ste_pkt"
    elif a.ptm_energy:
        label = "ste_ptm"
    elif a.init_seq:
        label = "ste_scaf"
    elif a.contact_weight:
        label = f"ste_ct{a.contact_weight:g}"
    elif a.confidence_weight:
        label = f"ste_c{a.confidence_weight:g}"
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
