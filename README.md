# nise-grad

Design small-molecule-binding proteins by gradient descent: optimize the binder
sequence directly through a differentiable structure model (joltz / Boltz-2) and a
differentiable P(bind).

Reimplements the idea of NISE (Polizzi lab,
https://www.nature.com/articles/s41586-026-10670-w) with gradients instead of a
gradient-free selection loop. Not a fork.

## What works
- Differentiable P(bind): the gradient of the Boltz-2 affinity head flows to the
  (soft) binder sequence through the structure model
  (`scripts/spike_affinity_gradient.py`).
- Gradient optimization of the sequence, optionally regularized by a ProteinMPNN
  sequence log-likelihood (`src/nisegrad/optimize.py`).

## What doesn't (open problem)
Naive gradient ascent on P(bind) reward-hacks: it drives the sequence to a degenerate
hydrophobic string that maxes the affinity logit without being a real binder. A
ProteinMPNN log-likelihood regularizer keeps the sequence protein-like, but P(bind) then
does not improve — and ProteinMPNN is *ligand-blind*, so it cannot see the interaction the
oracle is gaming.

## Ligand-aware regularizer
`src/nisegrad/ligand_mpnn_reg.py` scores the soft binder sequence with **LigandMPNN**
([jlig_mpnn](https://github.com/ssiddhantsharma/jlig_mpnn), a JAX port), conditioning on the
predicted backbone *and* the ligand context. Its NLL is a drop-in `mpnn=` regularizer for
`optimize_pbind`, differentiable in the sequence. `src/nisegrad/boltz_ligand.py` maps a Boltz
feature dict + output into LigandMPNN inputs (verified against the real feature layout);
`scripts/optimize_pbind_ligandmpnn.py` runs it, with a geometry-gated sanity fold first.

### GPU finding: the oracle's structures are too rough for a geometry prior
A LigandMPNN prior needs a *physical* backbone. The sanity gate showed the differentiable
Boltz-2 oracle does not provide one in the regime this method runs in:

| sampling steps | backbone bonds found (of ~119) | coordinate span |
|---|---|---|
| 2 (what `optimize_pbind` uses) | ~0 — coords are ±4000 Å noise | noise |
| 25 | 10 | ±28 Å but clashing |
| 50 | 19 | ±28 Å but clashing |

Full Boltz needs ~200 steps for physical geometry; that many is intractable for a per-step
gradient loop (memory + time). So conditioning on the predicted structure gives no useful
signal here — which is the same reason the ligand-*blind* ProteinMPNN prior above does not
help. **Ligand-awareness cannot fix the reward-hacking through structure when there is no
reliable structure to be aware of.** The extractor and geometry gate are kept: they make the
regularizer correct and drop-in the moment a folder returns physical coordinates (a
higher-step / cached-structure oracle, or a sequence-only ligand prior).

## Layout
- `src/nisegrad/oracle.py` — differentiable `P(bind)(sequence, ligand)`
- `src/nisegrad/optimize.py` — gradient ascent: P(bind), MPNN-regularized, or on-minus-off selectivity
- `src/nisegrad/ligand_mpnn_reg.py` — ligand-aware LigandMPNN sequence regularizer
- `scripts/` — the gradient check and the optimization run

## Install
Built on joltz (with the affinity head) + mosaic. GPU:

```
pip install -e .
# GPU jax that works with mosaic/joltz on CUDA 12 (newer nvidia libs break cuSPARSE):
pip install "jax[cuda12]==0.10.1"
pip install nvidia-cudnn-cu12==9.17.0.29 nvidia-cusolver-cu12==11.7.3.90 \
            nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```

Checkpoints (in `~/.boltz`, or point `NISEGRAD_BOLTZ_CACHE` at a dir with `ccd.pkl` +
`mols/`): `boltz2_conf.ckpt` (structure), `boltz2_aff.ckpt` (affinity head).

## Running efficiently
The cost is backprop through Boltz-2's diffusion structure module.
- `num_sampling_steps` (`PbindOracle`) is the main runtime/memory knob — 2–8 steps give
  a usable gradient; more is slower and OOMs sooner.
- Binder length drives O(N²) memory; ~30–40 aa at 2–4 sampling steps fits a 48 GB GPU.
- For P(bind) alone (no MPNN regularizer) you only need `structure_coordinates`, so the
  confidence module can be skipped for a further memory/speed cut.
- Larger designs need gradient checkpointing on the diffusion.
